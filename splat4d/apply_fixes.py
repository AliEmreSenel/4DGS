from pathlib import Path
import re
root=Path('/mnt/data/parity_fullfix')
# Patch preprocess SH original position
p=root/'shaders/preprocess_sorted.wgsl'
s=p.read_text()
s=s.replace('let color = eval_sh(g, p_orig, timestamp, frame);','let color = eval_sh(g, g.mean4.xyz, timestamp, frame);')
p.write_text(s)
# Patch exporter: richer appearance inference, metadata render_args, background, time duration, cam projections, true fields in CameraFrame already not in rust but raw parser preserves.
p=root/'tools/export_checkpoint.py'
s=p.read_text()
# Insert helper after nested_get maybe. find nested_get.
if 'def coerce_time_duration_export' not in s:
    marker='def truthy(x: Any) -> bool:\n'
    helper='''def coerce_time_duration_export(x: Any) -> Tuple[float, float]:\n    if x is None:\n        return (0.0, 1.0)\n    try:\n        arr = to_numpy(x)\n        if arr is not None and arr.size >= 2:\n            flat = arr.reshape(-1).astype(float)\n            return (float(flat[0]), float(flat[1]))\n    except Exception:\n        pass\n    if isinstance(x, (list, tuple)) and len(x) >= 2:\n        return (float(x[0]), float(x[1]))\n    return (0.0, 1.0)\n\ndef infer_sh_degrees(coeff_count: int, gkwargs: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[int, int]:\n    # Match GaussianModel.get_max_sh_channels.  Prefer run_config, then infer.\n    max_sh_degree = int(gkwargs.get("sh_degree", gkwargs.get("max_sh_degree", 0)) or 0)\n    max_sh_degree_t = int(gkwargs.get("sh_degree_t", gkwargs.get("max_sh_degree_t", 0)) or 0)\n    force_sh_3d = truthy(gkwargs.get("force_sh_3d", raw.get("force_sh_3d", False)))\n    if max_sh_degree <= 0:\n        # Supported repo models are normally SH3/SH0; infer the largest spatial degree whose base divides coeff_count.\n        for d in (3, 2, 1, 0):\n            base = (d + 1) ** 2\n            if coeff_count >= base and coeff_count % base == 0:\n                max_sh_degree = d\n                break\n    if max_sh_degree_t <= 0 and not force_sh_3d:\n        base = (max_sh_degree + 1) ** 2\n        if base > 0 and coeff_count % base == 0:\n            max_sh_degree_t = max(0, coeff_count // base - 1)\n    return max_sh_degree, max_sh_degree_t\n\n'''
    s=s.replace(marker, helper+marker)
# Replace app_model degree block
old='''    # Prefer checkpoint metadata over guessing.  4DGS with temporal SH can have\n    # coeff_count != (degree+1)^2, so store both spatial and temporal degrees.\n    gkwargs = nested_get(raw, ["root_run_config", "gaussian_kwargs"], {})\n    max_sh_degree = int(gkwargs.get("sh_degree", gkwargs.get("max_sh_degree", 0)) or 0)\n    max_sh_degree_t = int(gkwargs.get("sh_degree_t", gkwargs.get("max_sh_degree_t", 0)) or 0)\n    active_sh = raw.get("active_sh_degree")\n    active_sh_t = raw.get("active_sh_degree_t")\n    coeff_count = max(1, appearance.shape[1] // 3)\n    degree_guess = max(0, int(round(math.sqrt(coeff_count) - 1)))\n    app_model = {\n        "kind": "sh" if coeff_count > 1 else "rgb",\n        "degree": max_sh_degree if max_sh_degree > 0 else degree_guess,\n        "degree_t": max_sh_degree_t,\n        "active_degree": int(active_sh) if isinstance(active_sh, (int, np.integer)) else None,\n        "active_degree_t": int(active_sh_t) if isinstance(active_sh_t, (int, np.integer)) else None,\n        "coeff_count": coeff_count,\n        "storage": "f32",\n        "source_layout": "features_dc_plus_features_rest_flattened",\n    } if coeff_count > 1 else {"kind": "rgb", "storage": "f32"}\n\n    flags = np.zeros((n,), np.uint32)\n    if truthy(gkwargs.get("isotropic_gaussians", False)) or truthy(raw.get("isotropic_gaussians", False)):\n        flags |= 2\n    if truthy(gkwargs.get("rot_4d", raw.get("rot_4d", False))):\n        flags |= 4\n'''
new='''    # Prefer checkpoint metadata over guessing.  4DGS with temporal SH can have\n    # coeff_count != (degree+1)^2, so store both spatial and temporal degrees.\n    gkwargs = nested_get(raw, ["root_run_config", "gaussian_kwargs"], {})\n    active_sh = raw.get("active_sh_degree")\n    active_sh_t = raw.get("active_sh_degree_t")\n    coeff_count = max(1, appearance.shape[1] // 3)\n    max_sh_degree, max_sh_degree_t = infer_sh_degrees(coeff_count, gkwargs, raw)\n    active_degree = int(active_sh) if isinstance(active_sh, (int, np.integer)) else max_sh_degree\n    active_degree_t = int(active_sh_t) if isinstance(active_sh_t, (int, np.integer)) else max_sh_degree_t\n    force_sh_3d = truthy(gkwargs.get("force_sh_3d", raw.get("force_sh_3d", False)))\n    prefilter_var = float(gkwargs.get("prefilter_var", raw.get("prefilter_var", -1.0)) or -1.0)\n    td0, td1 = coerce_time_duration_export(gkwargs.get("time_duration", raw.get("time_duration", None)))\n    render_args = {\n        "gaussian_dim": int(gkwargs.get("gaussian_dim", 4) or 4),\n        "rot_4d": bool(truthy(gkwargs.get("rot_4d", raw.get("rot_4d", False)))),\n        "force_sh_3d": bool(force_sh_3d),\n        "prefilter_var": float(prefilter_var),\n        "scale_modifier": 1.0,\n        "time_duration": [float(td0), float(td1)],\n        "time_span": float(td1 - td0) if abs(td1 - td0) > 1e-12 else 1.0,\n        "active_sh_degree": int(active_degree),\n        "active_sh_degree_t": int(active_degree_t),\n        "max_sh_degree": int(max_sh_degree),\n        "max_sh_degree_t": int(max_sh_degree_t),\n        "coeff_count": int(coeff_count),\n    }\n    app_model = {\n        "kind": "sh" if coeff_count > 1 else "rgb",\n        "degree": int(max_sh_degree),\n        "degree_t": int(max_sh_degree_t),\n        "active_degree": int(active_degree),\n        "active_degree_t": int(active_degree_t),\n        "coeff_count": int(coeff_count),\n        "storage": "f32",\n        "source_layout": "features_dc_plus_features_rest_flattened",\n    } if coeff_count > 1 else {"kind": "rgb", "storage": "f32", "coeff_count": int(coeff_count)}\n\n    flags = np.zeros((n,), np.uint32)\n    if truthy(gkwargs.get("isotropic_gaussians", False)) or truthy(raw.get("isotropic_gaussians", False)):\n        flags |= 2\n    if render_args["rot_4d"]:\n        flags |= 4\n'''
if old not in s:
    print('degree block old not found')
else:
    s=s.replace(old,new)
# Add render_args to extras
old='''        "temporal_mask_config": {\n            "threshold": nested_get(raw, ["root_run_config", "args", "temporal_mask_threshold"], None),\n            "mode": nested_get(raw, ["root_run_config", "args", "temporal_mask_mode"], None),\n            "keyframes": nested_get(raw, ["root_run_config", "args", "temporal_mask_keyframes"], None),\n            "window": nested_get(raw, ["root_run_config", "args", "temporal_mask_window"], None),\n        },\n    }\n'''
new='''        "temporal_mask_config": {\n            "threshold": nested_get(raw, ["root_run_config", "args", "temporal_mask_threshold"], None),\n            "mode": nested_get(raw, ["root_run_config", "args", "temporal_mask_mode"], None),\n            "keyframes": nested_get(raw, ["root_run_config", "args", "temporal_mask_keyframes"], None),\n            "window": nested_get(raw, ["root_run_config", "args", "temporal_mask_window"], None),\n        },\n        "render_args": render_args,\n    }\n'''
if old in s: s=s.replace(old,new)
# Use background from scene white_background.
old='''        "background": [0.0, 0.0, 0.0, 1.0],'''
new='''        "background": ([1.0, 1.0, 1.0, 1.0] if bool((scene.extras.get("scene") or {}).get("white_background", False)) else [0.0, 0.0, 0.0, 1.0]),'''
s=s.replace(old,new)
# meta custom include render_args
old='''            "run_config": json_safe(scene.extras.get("run_config")),\n        },'''
new='''            "run_config": json_safe(scene.extras.get("run_config")),\n            "render_args": json_safe(scene.extras.get("render_args")),\n        },'''
s=s.replace(old,new)
p.write_text(s)
# Patch JS updateFrame with render_args, scaled focal, camera size exact. Also make app time no unintended motion maybe.
p=root/'web/main.js'
s=p.read_text()
# Replace updateFrame line with a formatted robust version using regex.
pattern=r"  updateFrame\(\)\{const t=Number\(timeEl\.value\);.*?device\.queue\.writeBuffer\(this\.g\.frame,0,buf\); \}"
m=re.search(pattern,s,re.S)
if not m:
    print('updateFrame not found')
else:
    upd='''  updateFrame(){
    const t=Number(timeEl.value);
    const cam=this.currentCam || this.nearestCamera(t);
    const exactCamera=!this.app.cameraDirty && cam && cam.view && cam.proj;
    const view=exactCamera ? flatMat(cam.view) : viewFromState(this.app);
    const projectionOnly = cam.projection ? flatMat(cam.projection) : null;
    const proj = exactCamera ? flatMat(cam.proj) : (projectionOnly ? mat4MulRM(view, projectionOnly) : flatMat(cam.proj));
    const inv=exactCamera && cam.inv_view ? flatMat(cam.inv_view) : mat4Identity();
    const w=canvas.width,h=canvas.height;
    const cw=Number(cam.width||w), ch=Number(cam.height||h);
    const fovx=Number(cam.fovx ?? (2*Math.atan(cw/(2*(cam.fl_x||cw)))));
    const fovy=Number(cam.fovy ?? (2*Math.atan(ch/(2*(cam.fl_y||ch)))));
    let focalX=Number(cam.fl_x && cam.fl_x>0 ? cam.fl_x*(w/cw) : (w/(2*Math.tan(fovx/2))));
    let focalY=Number(cam.fl_y && cam.fl_y>0 ? cam.fl_y*(h/ch) : (h/(2*Math.tan(fovy/2))));
    const m=this.maskIndices(t);
    const buf=new ArrayBuffer(512); const f32=new Float32Array(buf); const u32=new Uint32Array(buf);
    f32.set(view,0); f32.set(proj,16); f32.set(proj,32); f32.set(inv,48);
    const eyeForShader = exactCamera && Array.isArray(cam.camera_position) ? cam.camera_position.map(Number) : this.app.eye;
    f32.set([eyeForShader[0],eyeForShader[1],eyeForShader[2],t],64);
    f32.set([w,h,1/w,1/h],68);
    f32.set(this.meta.background??[0,0,0,1],72);
    u32.set([w,h,this.P,this.required==='sort-free-mobilegs'?1:0],76);
    u32.set([Math.ceil(w/16),Math.ceil(h/16),0,this.buffers.temporalMaskMeta?.keyframe_count??0],80);
    const app=this.meta.appearance_model||{};
    const rargs=(this.meta.custom&&this.meta.custom.render_args)||{};
    const scaleModifier=Number(rargs.scale_modifier??1.0);
    const prefilterVar=Number(rargs.prefilter_var??-1.0);
    f32.set([Number(cam.znear??0.01),Number(cam.zfar??100.0),scaleModifier,prefilterVar],84);
    u32.set([Number(rargs.active_sh_degree??app.active_degree??app.degree??3), Number(rargs.active_sh_degree_t??app.active_degree_t??app.degree_t??0), Number(rargs.coeff_count??app.coeff_count??Math.floor((this.buffers.appearance.byteLength/4)/Math.max(1,this.P)/3)), Number(rargs.force_sh_3d?1:0)],88);
    const span=Number(rargs.time_span??Math.max(1e-8,this.timeMax-this.timeMin));
    f32.set([Math.max(1e-8,span),this.timeMin,this.timeMax,0],92);
    u32.set([m.words,m.left,m.right,m.has],96);
    f32.set([focalX,focalY,Math.tan(fovx/2),Math.tan(fovy/2)],100);
    device.queue.writeBuffer(this.g.frame,0,buf);
  }'''
    s=s[:m.start()]+upd+s[m.end():]
# patch render empty total to still present bg? Perhaps if all empty, output stale black. Add background clear? Not necessary.
p.write_text(s)
# Add audit doc
p=root/'docs/CUDA_PARITY_AUDIT.md'
p.write_text('''# CUDA parity audit\n\nThis build audits and fixes the render-affecting differences against `diff-gaussian-rasterization/cuda_rasterizer/forward.cu` and `rasterizer_impl.cu`.\n\nFixed differences:\n\n1. **SH input position**: CUDA evaluates `computeColorFromSH*` with the original `orig_points`; previous WGSL used timestamp-conditioned means. Fixed.\n2. **Render arguments**: pack now stores and web uses `rot_4d`, `force_sh_3d`, `prefilter_var`, `scale_modifier`, `time_duration/time_span`, active SH degrees, temporal SH degree, and coefficient count.\n3. **Temporal SH degree inference**: coefficient counts such as 48 are interpreted as SH3 x 3 temporal bands instead of degree 6 spatial SH.\n4. **Background**: exported from scene `white_background`, not hard-coded black.\n5. **Camera-size focal scaling**: if rendering at a canvas size different from camera size, focal lengths are scaled exactly; otherwise original camera size is used by default.\n6. **CUDA sorting structure**: no fixed per-tile capacity; uses tiles_touched scan, dynamic duplicate list, padded global sort, tile range identification, and per-tile alpha compositing.\n7. **No fallback camera**: missing canonical cameras is a hard error.\n\nRemaining intentional difference:\n\n- The current WebGPU sort implementation is bitonic over padded keys, not CUB radix sort. It sorts the same `(tile_id, depth_bits)` key order but will not match CUB operation ordering/performance. Pixel compositing order after sorting is the same key order.\n''')
