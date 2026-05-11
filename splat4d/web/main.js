import init, { WasmScene } from './pkg/splat_wasm.js';

const logEl = document.getElementById('log');
const canvas = document.getElementById('view');
const fileEl = document.getElementById('file');
const contractEl = document.getElementById('renderContract');
const timeEl = document.getElementById('time');
const splitEl = document.getElementById('split');
const playEl = document.getElementById('play');
const prevCamEl = document.getElementById('prevCam');
const nextCamEl = document.getElementById('nextCam');
const captureEl = document.getElementById('capture');
const infoEl = document.getElementById('infoBtn');
const fullscreenEl = document.getElementById('fullscreen');
const overlayEl = document.getElementById('overlay');
const modelSelectEl = document.getElementById('modelSelect');
const loadModelEl = document.getElementById('loadModel');
const refreshModelsEl = document.getElementById('refreshModels');

function log(...msg) { logEl.textContent += msg.join(' ') + '\n'; }
function fatal(msg) { log('ERROR:', msg); throw new Error(msg); }
function clamp(x,a,b){return Math.max(a,Math.min(b,x));}
function wrapTime(t,t0,t1){const span=t1-t0; if(!Number.isFinite(span)||Math.abs(span)<1e-8)return t0; return t0 + ((((t-t0)%span)+span)%span);}
function add(a,b){return [a[0]+b[0],a[1]+b[1],a[2]+b[2]];} function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];} function mul(a,s){return [a[0]*s,a[1]*s,a[2]*s];} function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];} function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];} function norm(a){const n=Math.max(Math.hypot(a[0],a[1],a[2]),1e-20); return [a[0]/n,a[1]/n,a[2]/n];}
function rotateVec(v,axis,ang){const u=norm(axis), c=Math.cos(ang), s=Math.sin(ang); return add(add(mul(v,c),mul(cross(u,v),s)),mul(u,dot(u,v)*(1-c)));}
function mat4Identity(){return [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];}
function flatMat(m){if(Array.isArray(m)&&m.length===16)return m.map(Number); if(Array.isArray(m)&&m.length===4&&Array.isArray(m[0]))return m.flat().map(Number); return mat4Identity();}
function mat4MulCM(a,b){const o=new Array(16).fill(0); for(let c=0;c<4;c++)for(let r=0;r<4;r++){let v=0; for(let k=0;k<4;k++)v+=a[k*4+r]*b[c*4+k]; o[c*4+r]=v;} return o;}

function mat4MulRM(a,b){const o=new Array(16).fill(0); for(let r=0;r<4;r++)for(let c=0;c<4;c++){let v=0; for(let k=0;k<4;k++)v+=a[r*4+k]*b[k*4+c]; o[r*4+c]=v;} return o;}
function lookAtPlusZ(eye,target,up=[0,1,0]){const f=norm(sub(target,eye)), r=norm(cross(up,f)), u=cross(f,r); return [r[0],u[0],f[0],0,r[1],u[1],f[1],0,r[2],u[2],f[2],0,-dot(r,eye),-dot(u,eye),-dot(f,eye),1];}
function cameraFromMatrix(cam){const inv=cam.inv_view?flatMat(cam.inv_view):null; if(inv){const right=[inv[0],inv[1],inv[2]]; const down=[inv[4],inv[5],inv[6]]; const forward=[inv[8],inv[9],inv[10]]; const center=[inv[12],inv[13],inv[14]]; return {eye:center, orientation:{right:norm(right),up:norm(mul(down,-1)),forward:norm(forward)}};} const center=Array.isArray(cam.camera_position)?cam.camera_position.map(Number):[0,0,0]; return {eye:center, orientation:{right:[1,0,0],up:[0,1,0],forward:[0,0,1]}};}
function viewFromState(app){const o=normalizeOrientation(app.orientation); const down=mul(o.up,-1); return [o.right[0],down[0],o.forward[0],0,o.right[1],down[1],o.forward[1],0,o.right[2],down[2],o.forward[2],0,-dot(o.right,app.eye),-dot(down,app.eye),-dot(o.forward,app.eye),1];}
function normalizeOrientation(o){let f=norm(o.forward), r=norm(o.right); let u=norm(cross(f,r)); r=norm(cross(u,f)); return {right:r,up:u,forward:f};}
function nextPow2(x){let p=1; while(p<x)p*=2; return p;}
function u32Buffer(...v){const a=new Uint32Array(4); for(let i=0;i<v.length&&i<4;i++)a[i]=v[i]>>>0; return a;}
function f32Buffer(...v){const a=new Float32Array(4); for(let i=0;i<v.length&&i<4;i++)a[i]=Number(v[i]); return a;}
function readU64LE(bytes, off){const lo=bytes[off]|(bytes[off+1]<<8)|(bytes[off+2]<<16)|(bytes[off+3]<<24); const hi=bytes[off+4]|(bytes[off+5]<<8)|(bytes[off+6]<<16)|(bytes[off+7]<<24); return Number((BigInt(hi>>>0)<<32n)|BigInt(lo>>>0));}
function tag(bytes){return String.fromCharCode(bytes[0],bytes[1],bytes[2],bytes[3]);}
async function readSlice(file, off, len){return new Uint8Array(await file.slice(off, off+len).arrayBuffer());}

function isProbablyWebGPUAvailable() {
  return !!navigator.gpu;
}
function webgpuUnavailableMessage() {
  const secure = window.isSecureContext ? 'secure' : 'NOT secure';
  const ua = navigator.userAgent;
  return `WebGPU required: navigator.gpu is not available. Context is ${secure}. URL=${location.href}. On iPhone/iPad, use Safari with WebGPU support enabled and serve this page over HTTPS from the device-visible URL; http://<LAN-IP>:8080 is not a secure context. UA=${ua}`;
}
async function discoverRootModels() {
  const found = new Map();
  const add = (name, url) => {
    if (!name) return;
    if (!/\.(splat4dpack|s4dp)$/i.test(name)) return;
    found.set(name, url || `/${encodeURIComponent(name)}`);
  };
  try {
    const r = await fetch(`/models.json?v=${Date.now()}`, { cache: 'no-store' });
    if (r.ok) {
      const data = await r.json();
      const arr = Array.isArray(data) ? data : (Array.isArray(data.models) ? data.models : []);
      for (const item of arr) {
        if (typeof item === 'string') add(item.split('/').pop(), item.startsWith('/') ? item : `/${item}`);
        else if (item && typeof item === 'object') add(item.name || String(item.url || '').split('/').pop(), item.url || `/${item.name}`);
      }
    }
  } catch (_) {}
  try {
    const r = await fetch(`/?v=${Date.now()}`, { cache: 'no-store' });
    if (r.ok) {
      const html = await r.text();
      const doc = new DOMParser().parseFromString(html, 'text/html');
      for (const a of doc.querySelectorAll('a[href]')) {
        const href = a.getAttribute('href');
        if (!href) continue;
        const clean = decodeURIComponent(href.split('?')[0].replace(/^\//, ''));
        add(clean.split('/').pop(), href.startsWith('/') ? href : `/${href}`);
      }
    }
  } catch (_) {}
  return [...found.entries()].map(([name, url]) => ({ name, url })).sort((a,b)=>a.name.localeCompare(b.name));
}
async function refreshModelList() {
  if (!modelSelectEl) return;
  modelSelectEl.innerHTML = '<option value="">scanning / ...</option>';
  const models = await discoverRootModels();
  modelSelectEl.innerHTML = '';
  if (!models.length) {
    const opt = document.createElement('option');
    opt.value = ''; opt.textContent = 'no .splat4dpack files found in /';
    modelSelectEl.appendChild(opt);
    return;
  }
  for (const m of models) {
    const opt = document.createElement('option');
    opt.value = m.url; opt.textContent = m.name;
    modelSelectEl.appendChild(opt);
  }
}
async function loadPackFromUrl(url) {
  if (!url) return;
  logEl.textContent = '';
  log('fetching', url);
  const r = await fetch(`${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`, { cache: 'no-store' });
  if (!r.ok) fatal(`failed to load ${url}: ${r.status} ${r.statusText}`);
  const blob = await r.blob();
  const file = new File([blob], url.split('/').pop() || 'model.splat4dpack');
  const pack = await parsePackFile(file);
  await renderer.load(pack);
}


async function parsePackFile(file){
  const head=await readSlice(file,0,12); const magic=String.fromCharCode(...head.slice(0,8));
  if(magic!=='S4DPK3\0\0'){
    log('gzip or legacy pack detected; falling back to WASM decode (large gzip packs are not supported without high memory).');
    const scene=WasmScene.fromBytes(new Uint8Array(await file.arrayBuffer()));
    const meta=JSON.parse(scene.metaJson()); const buffers=scene.binaryBuffers();
    return {meta,buffers,gaussianCount:scene.gaussianCount(),requiredRenderType:scene.requiredRenderType(),allowedRenderTypes:Array.from(scene.allowedRenderTypes()),hasSortFreeMlp:scene.hasSortFreeMlp()};
  }
  const version = head[8] | (head[9]<<8) | (head[10]<<16) | (head[11]<<24); if(version!==3) fatal(`unsupported pack version ${version}`);
  let off=12; let meta=null; const buffers={gaussians:new Uint8Array(0),appearance:new Uint8Array(0),keyframes:new Uint8Array(0),maskWords:new Uint8Array(0),mlpWeights:new Uint8Array(0),aux:new Uint8Array(0),mlpMeta:null,temporalMaskMeta:null};
  while(off<file.size){const hdr=await readSlice(file,off,12); const t=tag(hdr); const len=readU64LE(hdr,4); off+=12; if(t==='META'){meta=JSON.parse(new TextDecoder().decode(await readSlice(file,off,len)));} else if(t==='GAUS'){buffers.gaussians=await readSlice(file,off,len);} else if(t==='APPR'){buffers.appearance=await readSlice(file,off,len);} else if(t==='KEYF'){buffers.keyframes=await readSlice(file,off,len);} else if(t==='MASK'){buffers.maskWords=await readSlice(file,off,len);} else if(t==='MLPW'){buffers.mlpWeights=await readSlice(file,off,len);} else if(t==='MLPM'){buffers.mlpMeta=JSON.parse(new TextDecoder().decode(await readSlice(file,off,len)));} else if(t==='TMSK'){buffers.temporalMaskMeta=JSON.parse(new TextDecoder().decode(await readSlice(file,off,len)));} else if(t==='AUX '){buffers.aux=await readSlice(file,off,len);} off+=len;}
  if(!meta) fatal('pack missing META');
  return {meta,buffers,gaussianCount:meta.gaussian_count,requiredRenderType:meta.render_policy.required_render_type,allowedRenderTypes:meta.render_policy.allowed_render_types,hasSortFreeMlp:meta.has_sortfree_mlp};
}

await init();
if(!isProbablyWebGPUAvailable()) fatal(webgpuUnavailableMessage());
const params=new URLSearchParams(location.search);
const cfg={width:Number(params.get('width')||1280),height:Number(params.get('height')||720),split:params.get('split')||'test',startCamera:Number(params.get('start_camera')||0),moveSpeed:Number(params.get('move_speed')||1.0),boost:Number(params.get('boost')||4.0),mouseSensitivity:Number(params.get('mouse_sensitivity')||0.01),rollSpeed:Number(params.get('roll_speed')||1.5),timeSpeed:Number(params.get('time_speed')||0.25),showInfo:params.get('show_info')==='1'};
canvas.width=cfg.width; canvas.height=cfg.height; canvas.tabIndex=0; splitEl.value=cfg.split;

const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}); if(!adapter) fatal('No WebGPU adapter.');
const req={}; for(const k of ['maxStorageBufferBindingSize','maxBufferSize']) if(Number.isFinite(adapter.limits[k])) req[k]=adapter.limits[k];
const device=await adapter.requestDevice({requiredLimits:req});
device.addEventListener('uncapturederror',e=>log('WebGPU error:', e.error?.message??e.message??e));
log('WebGPU limits:', `maxStorageBufferBindingSize=${device.limits.maxStorageBufferBindingSize}`, `maxBufferSize=${device.limits.maxBufferSize}`);
const context=canvas.getContext('webgpu'); const format=navigator.gpu.getPreferredCanvasFormat(); context.configure({device,format,alphaMode:'opaque'});

const shaderCache=new Map();
async function shader(path){if(shaderCache.has(path))return shaderCache.get(path); let txt=await fetch(`../shaders/${path}?v=${Date.now()}`,{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`${path} ${r.status}`); return r.text();}); txt=await resolveIncludes(txt); shaderCache.set(path,txt); return txt;}
async function resolveIncludes(txt){const re=/^#include\s+"(.+)"\s*$/gm; let m; while((m=re.exec(txt))){const inc=await shader(m[1]); txt=txt.replace(m[0],inc); re.lastIndex=0;} return txt;}
function makeBuffer(bytes, usage){const size=Math.max(4,(bytes.byteLength+3)&~3); const b=device.createBuffer({size,usage:usage|GPUBufferUsage.COPY_DST}); if(bytes.byteLength) device.queue.writeBuffer(b,0,bytes); return b;}
function makeEmpty(size, usage){return device.createBuffer({size:Math.max(4,(size+3)&~3),usage});}
function uniformU32(...v){const b=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}); device.queue.writeBuffer(b,0,u32Buffer(...v)); return b;}

class Renderer{
  constructor(){this.scene=null; this.keys=new Set(); this.ready=false; this.rendering=false; this.pending=false; this.tmpBuffers=[]; this.instanceCapacity=0; this.rangesTileCount=0; this.lastSubmittedAt=0; this.minFrameMs=Number(params.get('min_frame_ms')||0); this.app={paused:true,showInfo:cfg.showInfo,moveSpeed:cfg.moveSpeed,boost:cfg.boost,eye:[0,0,0],orientation:{right:[1,0,0],up:[0,1,0],forward:[0,0,1]},cameraDirty:false};}
  async initPipelines(){
    const mk=async path=>device.createComputePipeline({layout:'auto',compute:{module:device.createShaderModule({code:await shader(path)}),entryPoint:'main'}});
    this.pPre=await mk('preprocess_sorted.wgsl'); this.pScan=await mk('scan_step.wgsl'); this.pClearInst=await mk('clear_instances.wgsl'); this.pDup=await mk('duplicate_with_keys.wgsl'); this.pSort=await mk('bitonic_sort_pairs.wgsl'); this.pClearRanges=await mk('clear_ranges.wgsl'); this.pIdentify=await mk('identify_tile_ranges.wgsl'); this.pRender=await mk('render_sorted_cuda_parity.wgsl'); this.pRenderSF=await mk('render_sortfree_cuda_parity.wgsl');
    const pm=device.createShaderModule({code:await shader('present.wgsl')}); this.pPresent=device.createRenderPipeline({layout:'auto',vertex:{module:pm,entryPoint:'vs'},fragment:{module:pm,entryPoint:'fs',targets:[{format}]},primitive:{topology:'triangle-list'}});
  }
  resize(w,h){canvas.width=w; canvas.height=h; this.output=device.createTexture({size:[w,h],format:'rgba16float',usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}); this.outputView=this.output.createView(); this.bPresent=device.createBindGroup({layout:this.pPresent.getBindGroupLayout(0),entries:[{binding:0,resource:this.outputView}]});}
  async load(pack){
    this.ready=false; if(!this.pPre) await this.initPipelines(); this.scene=pack; this.meta=pack.meta; this.buffers=pack.buffers; this.required=pack.requiredRenderType; this.P=pack.gaussianCount; this.cameras=Array.isArray(this.meta.cameras)?this.meta.cameras:[]; if(!this.cameras.length) fatal('Pack has no canonical cameras. Re-export the checkpoint; reference CUDA parity refuses fallback cameras.');
    contractEl.textContent=`locked render contract: ${this.required}`; log('loaded',this.meta.name,`${this.P} gaussians`); log('render contract:',this.required); log('cameras:',this.cameras.length);
    const backend=this.meta.custom?.export_backend || this.meta.custom?.render_args?.export_backend || 'unknown';
    log('export backend:', backend);
    if(backend!=='repo-gaussian-model-getters') log('WARNING: this pack was not exported through the repo GaussianModel getters; re-export with this build for CUDA parity.');
    if(this.required==='sort-free-mobilegs' && !pack.hasSortFreeMlp) fatal('sort-free-mobilegs pack missing required MLP.');
    const cams=this.cameraList(); const cam=cams[clamp(cfg.startCamera,0,Math.max(0,cams.length-1))]||this.cameras[0]; this.snapToCamera(cam);
    const ts=this.cameras.map(c=>Number(c.timestamp??0)); this.timeMin=params.has('time_start')?Number(params.get('time_start')):Math.min(...ts); this.timeMax=params.has('time_end')?Number(params.get('time_end')):Math.max(...ts); timeEl.min=this.timeMin; timeEl.max=this.timeMax; timeEl.step=(this.timeMax-this.timeMin)/1000; timeEl.value=clamp(Number(cam.timestamp??this.timeMin),this.timeMin,this.timeMax);
    this.g={}; this.g.gaussians=makeBuffer(this.buffers.gaussians,GPUBufferUsage.STORAGE); this.g.appearance=makeBuffer(this.buffers.appearance,GPUBufferUsage.STORAGE); this.g.keyframes=makeBuffer(this.buffers.keyframes,GPUBufferUsage.STORAGE); this.g.masks=makeBuffer(this.buffers.maskWords,GPUBufferUsage.STORAGE); this.g.prep=makeEmpty(this.P*64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST); this.g.tiles=makeEmpty(this.P*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST); this.g.scanA=makeEmpty(this.P*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST); this.g.scanB=makeEmpty(this.P*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST); this.g.frame=device.createBuffer({size:512,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.resize(Number(cam.width||cfg.width),Number(cam.height||cfg.height)); this.ready=true; await this.render(); canvas.focus({preventScroll:true});
  }
  cameraList(){const sp=splitEl.value||cfg.split; if(sp==='all')return this.cameras; const list=this.cameras.filter(c=>c.split===sp); return list.length?list:this.cameras;}
  snapToCamera(cam){const st=cameraFromMatrix(cam); this.app.eye=st.eye; this.app.orientation=st.orientation; this.currentCam=cam; this.app.cameraDirty=false; if(cam.width&&cam.height) this.resize(Number(cam.width),Number(cam.height));}
  nearestCamera(t){let best=this.cameras[0],bd=Math.abs(Number(best.timestamp??0)-t); for(const c of this.cameras){const d=Math.abs(Number(c.timestamp??0)-t); if(d<bd){best=c;bd=d;}} return best;}
  maskIndices(t){const tm=this.buffers.temporalMaskMeta; if(!tm||!this.buffers.keyframes?.byteLength) return {words:0,left:0,right:0,has:0}; const keys=new Float32Array(this.buffers.keyframes.buffer,this.buffers.keyframes.byteOffset,this.buffers.keyframes.byteLength/4); let right=0; while(right<keys.length&&keys[right]<t)right++; let left=Math.max(0,Math.min(keys.length-1,right-1)); right=Math.max(0,Math.min(keys.length-1,right)); return {words:tm.words_per_mask||Math.ceil(this.P/32),left,right,has:1};}
  updateFrame(){
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
    // CUDA rasterizer computes focal_x/y only from tanfovx/tanfovy:
    //   focal_x = width / (2 * tan_fovx)
    //   focal_y = height / (2 * tan_fovy)
    // Do not substitute fl_x/fl_y here; those are only used to build the
    // projection matrix for center-shift cameras.
    let focalX=Number(w/(2*Math.tan(fovx/2)));
    let focalY=Number(h/(2*Math.tan(fovy/2)));
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
  }
  bind(pipeline, entries){return device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries});}
  tempUniformU32(...v){const b=uniformU32(...v); this.tmpBuffers.push(b); return b;}
  destroyBuffersLater(buffers){const list=buffers.filter(Boolean); if(!list.length)return; device.queue.onSubmittedWorkDone().then(()=>{for(const b of list){try{b.destroy();}catch(_){}}});}
  ensureInstanceBuffers(padded,tileCount){const keyBytes=padded*4; if(!this.g.keysHi || padded>this.instanceCapacity){const old=[this.g.keysHi,this.g.keysLo,this.g.values]; this.g.keysHi=makeEmpty(keyBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST); this.g.keysLo=makeEmpty(keyBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST); this.g.values=makeEmpty(keyBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST); this.instanceCapacity=padded; this.destroyBuffersLater(old);} if(!this.g.ranges || tileCount!==this.rangesTileCount){const old=this.g.ranges; this.g.ranges=makeEmpty(tileCount*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST); this.rangesTileCount=tileCount; this.destroyBuffersLater([old]);}}
  pass(encoder,pipeline,bg,x,y=1){const p=encoder.beginComputePass(); p.setPipeline(pipeline); p.setBindGroup(0,bg); let xx=Math.max(1,Math.ceil(x)); let yy=Math.max(1,Math.ceil(y)); if(yy===1&&xx>65535){yy=Math.ceil(xx/65535); xx=65535;} p.dispatchWorkgroups(xx,yy); p.end();}
  scanBind(inb,outb,n,offset){const ub=this.tempUniformU32(n,offset,0,0); return this.bind(this.pScan,[{binding:0,resource:{buffer:ub}},{binding:1,resource:{buffer:inb}},{binding:2,resource:{buffer:outb}}]);}
  async scanTiles(){let enc=device.createCommandEncoder(); const preBg=this.bind(this.pPre,[{binding:0,resource:{buffer:this.g.frame}},{binding:1,resource:{buffer:this.g.gaussians}},{binding:2,resource:{buffer:this.g.appearance}},{binding:4,resource:{buffer:this.g.masks}},{binding:5,resource:{buffer:this.g.prep}},{binding:6,resource:{buffer:this.g.tiles}}]); this.pass(enc,this.pPre,preBg,Math.ceil(this.P/256)); enc.copyBufferToBuffer(this.g.tiles,0,this.g.scanA,0,this.P*4); let input=this.g.scanA,output=this.g.scanB; for(let off=1;off<this.P;off<<=1){this.pass(enc,this.pScan,this.scanBind(input,output,this.P,off),Math.ceil(this.P/256)); [input,output]=[output,input];} const rb=device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}); enc.copyBufferToBuffer(input,(this.P-1)*4,rb,0,4); device.queue.submit([enc.finish()]); await rb.mapAsync(GPUMapMode.READ); const total=new Uint32Array(rb.getMappedRange())[0]; rb.unmap(); rb.destroy(); const temps=this.tmpBuffers.splice(0); this.destroyBuffersLater(temps); this.offsets=input; return total;}
  async render(){if(!this.ready)return; const now=performance.now(); if(this.minFrameMs>0 && now-this.lastSubmittedAt<this.minFrameMs){this.pending=true;return;} if(this.rendering){this.pending=true;return;} this.rendering=true; this.tmpBuffers=[]; try{this.updateFrame(); const total=await this.scanTiles(); if(total===0){log('0 visible splat/tile instances'); this.rendering=false; return;} const padded=nextPow2(total); const tileCount=Math.ceil(canvas.width/16)*Math.ceil(canvas.height/16); this.ensureInstanceBuffers(padded,tileCount); let enc=device.createCommandEncoder(); this.pass(enc,this.pClearInst,this.bind(this.pClearInst,[{binding:0,resource:{buffer:this.tempUniformU32(padded,0,0,0)}},{binding:1,resource:{buffer:this.g.keysHi}},{binding:2,resource:{buffer:this.g.keysLo}},{binding:3,resource:{buffer:this.g.values}}]),Math.ceil(padded/256)); this.pass(enc,this.pDup,this.bind(this.pDup,[{binding:0,resource:{buffer:this.g.frame}},{binding:5,resource:{buffer:this.g.prep}},{binding:6,resource:{buffer:this.g.tiles}},{binding:7,resource:{buffer:this.offsets}},{binding:9,resource:{buffer:this.g.keysHi}},{binding:10,resource:{buffer:this.g.keysLo}},{binding:11,resource:{buffer:this.g.values}}]),Math.ceil(this.P/256)); for(let k=2;k<=padded;k<<=1){for(let j=k>>1;j>0;j>>=1){this.pass(enc,this.pSort,this.bind(this.pSort,[{binding:0,resource:{buffer:this.tempUniformU32(padded,k,j,0)}},{binding:1,resource:{buffer:this.g.keysHi}},{binding:2,resource:{buffer:this.g.keysLo}},{binding:3,resource:{buffer:this.g.values}}]),Math.ceil(padded/256));}} this.pass(enc,this.pClearRanges,this.bind(this.pClearRanges,[{binding:0,resource:{buffer:this.tempUniformU32(tileCount,0,0,0)}},{binding:1,resource:{buffer:this.g.ranges}}]),Math.ceil(tileCount/256)); this.pass(enc,this.pIdentify,this.bind(this.pIdentify,[{binding:0,resource:{buffer:this.tempUniformU32(total,tileCount,0,0)}},{binding:1,resource:{buffer:this.g.keysHi}},{binding:2,resource:{buffer:this.g.ranges}}]),Math.ceil(total/256)); const rpipe=this.required==='sort-free-mobilegs'?this.pRenderSF:this.pRender; this.pass(enc,rpipe,this.bind(rpipe,[{binding:0,resource:{buffer:this.g.frame}},{binding:5,resource:{buffer:this.g.prep}},{binding:11,resource:{buffer:this.g.values}},{binding:12,resource:{buffer:this.g.ranges}},{binding:13,resource:this.outputView}]),Math.ceil(canvas.width/16),Math.ceil(canvas.height/16)); const rp=enc.beginRenderPass({colorAttachments:[{view:context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:'clear',storeOp:'store'}]}); rp.setPipeline(this.pPresent); rp.setBindGroup(0,this.bPresent); rp.draw(3); rp.end(); device.queue.submit([enc.finish()]); this.lastSubmittedAt=performance.now(); const temps=this.tmpBuffers.splice(0); this.destroyBuffersLater(temps); this.updateOverlay(total); } finally{this.rendering=false; if(this.pending){this.pending=false; requestAnimationFrame(()=>this.render());}}}
  updateMotion(dt){let moved=false; const k=this.keys; const o=normalizeOrientation(this.app.orientation); const speed=this.app.moveSpeed*(k.has('ShiftLeft')||k.has('ShiftRight')?this.app.boost:1)*dt; const has=(...a)=>a.some(x=>k.has(x)); if(has('KeyW','ArrowUp')){this.app.eye=add(this.app.eye,mul(o.forward,speed)); moved=true;} if(has('KeyS','ArrowDown')){this.app.eye=sub(this.app.eye,mul(o.forward,speed)); moved=true;} if(has('KeyD','ArrowRight')){this.app.eye=add(this.app.eye,mul(o.right,speed)); moved=true;} if(has('KeyA','ArrowLeft')){this.app.eye=sub(this.app.eye,mul(o.right,speed)); moved=true;} if(has('Space')){this.app.eye=add(this.app.eye,mul(o.up,speed)); moved=true;} if(has('ControlLeft','ControlRight')){this.app.eye=sub(this.app.eye,mul(o.up,speed)); moved=true;} if(has('KeyR')){this.app.orientation=normalizeOrientation({right:rotateVec(o.right,o.forward,-cfg.rollSpeed*dt),up:rotateVec(o.up,o.forward,-cfg.rollSpeed*dt),forward:o.forward}); moved=true;} if(has('KeyT')){this.app.orientation=normalizeOrientation({right:rotateVec(o.right,o.forward,cfg.rollSpeed*dt),up:rotateVec(o.up,o.forward,cfg.rollSpeed*dt),forward:o.forward}); moved=true;} if(!this.app.paused){timeEl.value=wrapTime(Number(timeEl.value)+cfg.timeSpeed*dt,this.timeMin,this.timeMax); moved=true;} if(moved){this.app.cameraDirty=true; this.render();}}
  updateOverlay(total=0){if(!this.app.showInfo){overlayEl.style.display='none';return;} overlayEl.style.display='block'; const o=normalizeOrientation(this.app.orientation); overlayEl.textContent=`${this.required} | P=${this.P} instances=${total}\ntime ${Number(timeEl.value).toFixed(5)} [${this.timeMin.toFixed(3)},${this.timeMax.toFixed(3)}]\neye ${this.app.eye.map(x=>x.toFixed(3)).join(' ')}\nfwd ${o.forward.map(x=>x.toFixed(3)).join(' ')}`;}
}

const renderer=new Renderer();
fileEl.addEventListener('change',async()=>{const file=fileEl.files?.[0]; if(!file)return; logEl.textContent=''; try{const pack=await parsePackFile(file); await renderer.load(pack);}catch(e){fatal(e.message??String(e));}});
refreshModelsEl?.addEventListener('click',()=>refreshModelList());
loadModelEl?.addEventListener('click',()=>loadPackFromUrl(modelSelectEl.value).catch(e=>fatal(e.message??String(e))));
modelSelectEl?.addEventListener('change',()=>{ if(params.get('autoload_on_select')==='1') loadPackFromUrl(modelSelectEl.value).catch(e=>fatal(e.message??String(e))); });
refreshModelList().then(()=>{ const m=params.get('model'); if(m){ const url=m.startsWith('/')?m:`/${m}`; if(modelSelectEl){ for(const opt of modelSelectEl.options){ if(opt.value===url || opt.textContent===m) opt.selected=true; } } loadPackFromUrl(url).catch(e=>fatal(e.message??String(e))); } });
timeEl.addEventListener('input',()=>renderer.render());
splitEl.addEventListener('change',()=>{const cams=renderer.cameraList(); if(cams.length){renderer.snapToCamera(cams[0]); renderer.render();}});
prevCamEl.addEventListener('click',()=>{const cams=renderer.cameraList(); if(!cams.length)return; let i=cams.indexOf(renderer.currentCam); i=(i-1+cams.length)%cams.length; renderer.snapToCamera(cams[i]); timeEl.value=Number(cams[i].timestamp??timeEl.value); renderer.render();});
nextCamEl.addEventListener('click',()=>{const cams=renderer.cameraList(); if(!cams.length)return; let i=cams.indexOf(renderer.currentCam); i=(i+1)%cams.length; renderer.snapToCamera(cams[i]); timeEl.value=Number(cams[i].timestamp??timeEl.value); renderer.render();});
playEl.addEventListener('click',()=>{renderer.app.paused=!renderer.app.paused; playEl.textContent=renderer.app.paused?'play':'pause';});
infoEl.addEventListener('click',()=>{renderer.app.showInfo=!renderer.app.showInfo; infoEl.classList.toggle('active',renderer.app.showInfo); renderer.updateOverlay();});
captureEl.addEventListener('click',()=>{canvas.focus({preventScroll:true}); canvas.requestPointerLock();});
fullscreenEl.addEventListener('click',async()=>{canvas.focus({preventScroll:true}); if(!document.fullscreenElement)await canvas.parentElement.requestFullscreen(); else await document.exitFullscreen(); canvas.focus({preventScroll:true});});
document.addEventListener('pointerlockchange',()=>{captureEl.classList.toggle('active',document.pointerLockElement===canvas);});
canvas.addEventListener('click',()=>canvas.focus({preventScroll:true}));
window.addEventListener('keydown',e=>{const codes=['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space','ControlLeft','ControlRight','ShiftLeft','ShiftRight','KeyR','KeyT']; if(codes.includes(e.code)){e.preventDefault(); canvas.focus({preventScroll:true});} renderer.keys.add(e.code); if(e.code==='KeyP'){renderer.app.paused=!renderer.app.paused;} if(e.code==='Equal'||e.code==='NumpadAdd'){renderer.app.moveSpeed*=1.25; log('move speed',renderer.app.moveSpeed.toFixed(3));} if(e.code==='Minus'||e.code==='NumpadSubtract'){renderer.app.moveSpeed/=1.25; log('move speed',renderer.app.moveSpeed.toFixed(3));}}, {capture:true});
window.addEventListener('keyup',e=>renderer.keys.delete(e.code),{capture:true});
window.addEventListener('mousemove',e=>{if(document.pointerLockElement!==canvas||!renderer.scene)return; let o=normalizeOrientation(renderer.app.orientation); if(e.movementX){o={right:rotateVec(o.right,o.up,e.movementX*cfg.mouseSensitivity),up:o.up,forward:rotateVec(o.forward,o.up,e.movementX*cfg.mouseSensitivity)};} if(e.movementY){o=normalizeOrientation(o); o={right:o.right,up:rotateVec(o.up,o.right,e.movementY*cfg.mouseSensitivity),forward:rotateVec(o.forward,o.right,e.movementY*cfg.mouseSensitivity)};} renderer.app.orientation=normalizeOrientation(o); renderer.app.cameraDirty=true; renderer.render();});
document.addEventListener('fullscreenchange',()=>canvas.focus({preventScroll:true}));
let last=performance.now(); function tick(now){const dt=Math.min(0.1,(now-last)/1000); last=now; if(renderer.scene)renderer.updateMotion(dt); requestAnimationFrame(tick);} requestAnimationFrame(tick);
