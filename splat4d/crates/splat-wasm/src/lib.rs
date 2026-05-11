use js_sys::{Array, Object, Reflect, Uint8Array};
use splat_format::{ScenePackageV3, RenderType};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); }

#[wasm_bindgen]
pub struct WasmScene {
    scene: ScenePackageV3,
}

#[wasm_bindgen]
impl WasmScene {
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmScene, JsValue> {
        let scene = ScenePackageV3::decode(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmScene { scene })
    }

    #[wasm_bindgen(js_name = metaJson)]
    pub fn meta_json(&self) -> Result<String, JsValue> {
        serde_json::to_string_pretty(&self.scene.meta).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = defaultRenderType)]
    pub fn default_render_type(&self) -> String {
        render_type_to_str(&self.scene.meta.render_policy.default_render_type).to_string()
    }

    #[wasm_bindgen(js_name = referenceRenderType)]
    pub fn reference_render_type(&self) -> String {
        render_type_to_str(&self.scene.meta.render_policy.reference_render_type).to_string()
    }

    #[wasm_bindgen(js_name = requiredRenderType)]
    pub fn required_render_type(&self) -> String {
        render_type_to_str(&self.scene.meta.render_policy.required_render_type).to_string()
    }

    #[wasm_bindgen(js_name = allowedRenderTypes)]
    pub fn allowed_render_types(&self) -> Array {
        self.scene.meta.render_policy.allowed_render_types.iter().map(|t| JsValue::from_str(render_type_to_str(t))).collect()
    }

    #[wasm_bindgen(js_name = gaussianCount)]
    pub fn gaussian_count(&self) -> u32 { self.scene.meta.gaussian_count }

    #[wasm_bindgen(js_name = hasTemporalMasks)]
    pub fn has_temporal_masks(&self) -> bool { self.scene.meta.has_temporal_masks }

    #[wasm_bindgen(js_name = hasSortFreeMlp)]
    pub fn has_sortfree_mlp(&self) -> bool { self.scene.meta.has_sortfree_mlp }

    #[wasm_bindgen(js_name = binaryBuffers)]
    pub fn binary_buffers(&self) -> Result<Object, JsValue> {
        let out = Object::new();
        put_bytes(&out, "gaussians", bytemuck::cast_slice(&self.scene.gaussians))?;
        put_bytes(&out, "appearance", bytemuck::cast_slice(&self.scene.appearance))?;
        put_bytes(&out, "aux", bytemuck::cast_slice(&self.scene.aux))?;
        put_bytes(&out, "keyframes", bytemuck::cast_slice(&self.scene.keyframes))?;
        put_bytes(&out, "maskWords", bytemuck::cast_slice(&self.scene.mask_words))?;
        put_bytes(&out, "mlpWeights", bytemuck::cast_slice(&self.scene.mlp_weights))?;
        if let Some(m) = &self.scene.mlp_meta {
            Reflect::set(&out, &JsValue::from_str("mlpMeta"), &serde_wasm_bindgen::to_value(m)?)?;
        }
        if let Some(t) = &self.scene.temporal_mask_meta {
            Reflect::set(&out, &JsValue::from_str("temporalMaskMeta"), &serde_wasm_bindgen::to_value(t)?)?;
        }
        Ok(out)
    }
}

fn render_type_to_str(t: &RenderType) -> &'static str {
    match t {
        RenderType::SortedAlpha => "sorted-alpha",
        RenderType::SortFreeMobileGs => "sort-free-mobilegs",
        RenderType::SortFreeWeightedOit => "sort-free-weighted-oit",
        RenderType::WebglPreview => "webgl-preview",
    }
}

fn put_bytes(obj: &Object, key: &str, bytes: &[u8]) -> Result<(), JsValue> {
    let arr = Uint8Array::new_with_length(bytes.len() as u32);
    arr.copy_from(bytes);
    Reflect::set(obj, &JsValue::from_str(key), &arr)?;
    Ok(())
}

#[wasm_bindgen(js_name = supportedRenderTypes)]
pub fn supported_render_types() -> Array {
    ["pack-required-only"].iter().map(|s| JsValue::from_str(s)).collect()
}
