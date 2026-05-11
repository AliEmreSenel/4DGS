use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};

pub const MAGIC_V3: &[u8; 8] = b"S4DPK3\0\0";
pub const FORMAT_VERSION: u32 = 3;

pub mod flags {
    pub const STATIC_GAUSSIAN: u32 = 1 << 0;
    pub const ISOTROPIC: u32 = 1 << 1;
    pub const HAS_MLP_FEATURES: u32 = 1 << 2;
    pub const PRESERVED_COMPRESSED_SOURCE: u32 = 1 << 3;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct GaussianRecordV3 {
    pub mean4: [f32; 4],
    pub scale4: [f32; 4],
    pub q_left: [f32; 4],
    pub q_right: [f32; 4],
    pub opacity: f32,
    pub appearance_offset: u32,
    pub appearance_len: u32,
    pub aux_offset: u32,
    pub aux_len: u32,
    pub flags: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RenderType {
    SortedAlpha,
    #[serde(rename = "sort-free-mobilegs", alias = "sort-free-mobile-gs")]
    SortFreeMobileGs,
    SortFreeWeightedOit,
    WebglPreview,
}

impl Default for RenderType {
    fn default() -> Self { RenderType::SortedAlpha }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderPreset {
    pub render_type: RenderType,
    pub adaptive: bool,
    pub requires_explicit_url_parameter: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderPolicy {
    pub native_render_type: RenderType,
    pub default_render_type: RenderType,
    pub reference_render_type: RenderType,
    /// The one renderer that is semantically valid for this checkpoint.
    /// Sorted and sort-free checkpoints are different render contracts; this field is authoritative.
    #[serde(default)]
    pub required_render_type: RenderType,
    /// Renderers that are mathematically equivalent for this pack. In normal production packs this
    /// contains exactly one value. Debug/inspection renderers must be separate tools, not quality modes.
    #[serde(default)]
    pub allowed_render_types: Vec<RenderType>,
    /// Renderers that must never be selected for this pack.
    #[serde(default)]
    pub forbidden_render_types: Vec<RenderType>,
    #[serde(default)]
    pub render_type_locked: bool,
    pub allow_url_override: bool,
    #[serde(default)]
    pub quality_presets: BTreeMap<String, RenderPreset>,
}

impl Default for RenderPolicy {
    fn default() -> Self {
        Self {
            native_render_type: RenderType::SortedAlpha,
            default_render_type: RenderType::SortedAlpha,
            reference_render_type: RenderType::SortedAlpha,
            required_render_type: RenderType::SortedAlpha,
            allowed_render_types: vec![RenderType::SortedAlpha],
            forbidden_render_types: vec![RenderType::SortFreeMobileGs, RenderType::SortFreeWeightedOit, RenderType::WebglPreview],
            render_type_locked: true,
            allow_url_override: false,
            quality_presets: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum AppearanceModel {
    Rgb,
    Sh { degree: u8, coeff_count: u32, storage: TensorStorage },
    Spherindrical { sh_degree: u8, fourier_degree: u8, period: f32, coeff_count: u32, storage: TensorStorage },
    RgbWithAux { aux_dim: u32, storage: TensorStorage },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorStorage {
    F32,
    F16,
    U8Affine,
    U16Affine,
    Codebook,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraFrame {
    pub name: String,
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub timestamp: f32,
    pub width: u32,
    pub height: u32,
    pub split: String,
    #[serde(default)]
    pub fovx: Option<f32>,
    #[serde(default)]
    pub fovy: Option<f32>,
    #[serde(default)]
    pub cx: Option<f32>,
    #[serde(default)]
    pub cy: Option<f32>,
    #[serde(default)]
    pub fl_x: Option<f32>,
    #[serde(default)]
    pub fl_y: Option<f32>,
    #[serde(default)]
    pub znear: Option<f32>,
    #[serde(default)]
    pub zfar: Option<f32>,
}


fn deserialize_camera_frames_flexible<'de, D>(deserializer: D) -> std::result::Result<Vec<CameraFrame>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                if let Ok(frame) = serde_json::from_value::<CameraFrame>(item) {
                    out.push(frame);
                }
            }
            Ok(out)
        }
        // Older exporter builds wrote {train:[...], test:[...]} here. That raw camera metadata is
        // preserved in EXTR/custom; for compatibility do not reject those packs.
        serde_json::Value::Object(_) | serde_json::Value::Null => Ok(Vec::new()),
        _ => Ok(Vec::new()),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointProvenance {
    pub source_path: Option<String>,
    pub source_format: String,
    pub source_sha256: Option<String>,
    pub schema: String,
    pub source_keys: Vec<String>,
    pub preserved_original: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalMaskMeta {
    pub keyframe_count: u32,
    pub words_per_mask: u32,
    pub interpolation: String,
    pub mask_order: String,
    pub notes: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlpLayerMeta {
    pub name: String,
    pub in_dim: u32,
    pub out_dim: u32,
    pub weight_offset: u32,
    pub weight_len: u32,
    pub bias_offset: u32,
    pub bias_len: u32,
    pub activation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlpMeta {
    pub purpose: String,
    pub input_layout: Vec<String>,
    pub output_layout: Vec<String>,
    pub dtype: TensorStorage,
    pub layers: Vec<MlpLayerMeta>,
    pub custom_formula: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionMeta {
    pub source_was_compressed: bool,
    pub decoded_for_render: bool,
    pub schemes: Vec<String>,
    pub has_codebooks: bool,
    pub has_huffman: bool,
    pub has_rvq: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneMetaV3 {
    pub name: String,
    pub gaussian_count: u32,
    pub time_min: f32,
    pub time_max: f32,
    pub background: [f32; 4],
    pub appearance_model: AppearanceModel,
    pub render_policy: RenderPolicy,
    pub has_temporal_masks: bool,
    pub has_sortfree_mlp: bool,
    pub has_env_map: bool,
    pub has_compression_payloads: bool,
    pub provenance: CheckpointProvenance,
    pub compression: CompressionMeta,
    #[serde(default, deserialize_with = "deserialize_camera_frames_flexible")]
    pub cameras: Vec<CameraFrame>,
    pub custom: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    pub tag: [u8; 4],
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ScenePackageV3 {
    pub meta: SceneMetaV3,
    pub gaussians: Vec<GaussianRecordV3>,
    pub appearance: Vec<f32>,
    pub aux: Vec<f32>,
    pub keyframes: Vec<f32>,
    pub mask_words: Vec<u32>,
    pub temporal_mask_meta: Option<TemporalMaskMeta>,
    pub mlp_meta: Option<MlpMeta>,
    pub mlp_weights: Vec<f32>,
    pub extra_chunks: Vec<Chunk>,
}

fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) -> Result<()> {
    out.extend_from_slice(tag);
    out.write_u64::<LittleEndian>(payload.len() as u64)?;
    out.extend_from_slice(payload);
    Ok(())
}

fn read_chunk(cur: &mut Cursor<&[u8]>) -> Result<Option<Chunk>> {
    if cur.position() as usize >= cur.get_ref().len() { return Ok(None); }
    let mut tag = [0_u8; 4];
    cur.read_exact(&mut tag)?;
    let len = cur.read_u64::<LittleEndian>()? as usize;
    let pos = cur.position() as usize;
    let end = pos.checked_add(len).context("chunk length overflow")?;
    if end > cur.get_ref().len() { bail!("truncated chunk {:?}", std::str::from_utf8(&tag).unwrap_or("????")); }
    let payload = cur.get_ref()[pos..end].to_vec();
    cur.set_position(end as u64);
    Ok(Some(Chunk { tag, payload }))
}

impl ScenePackageV3 {
    pub fn encode(&self, gzip: bool) -> Result<Vec<u8>> {
        self.validate()?;
        let mut raw = Vec::new();
        raw.extend_from_slice(MAGIC_V3);
        raw.write_u32::<LittleEndian>(FORMAT_VERSION)?;
        write_chunk(&mut raw, b"META", &serde_json::to_vec_pretty(&self.meta)?)?;
        write_chunk(&mut raw, b"GAUS", bytemuck::cast_slice(&self.gaussians))?;
        write_chunk(&mut raw, b"APPR", bytemuck::cast_slice(&self.appearance))?;
        if !self.aux.is_empty() { write_chunk(&mut raw, b"AUX ", bytemuck::cast_slice(&self.aux))?; }
        if let Some(t) = &self.temporal_mask_meta {
            write_chunk(&mut raw, b"TMSK", &serde_json::to_vec_pretty(t)?)?;
            write_chunk(&mut raw, b"KEYF", bytemuck::cast_slice(&self.keyframes))?;
            write_chunk(&mut raw, b"MASK", bytemuck::cast_slice(&self.mask_words))?;
        }
        if let Some(m) = &self.mlp_meta {
            write_chunk(&mut raw, b"MLPM", &serde_json::to_vec_pretty(m)?)?;
            write_chunk(&mut raw, b"MLPW", bytemuck::cast_slice(&self.mlp_weights))?;
        }
        for ch in &self.extra_chunks { write_chunk(&mut raw, &ch.tag, &ch.payload)?; }
        if gzip {
            let mut gz = GzEncoder::new(Vec::new(), Compression::new(6));
            gz.write_all(&raw)?;
            Ok(gz.finish()?)
        } else { Ok(raw) }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut decoded = Vec::new();
        if bytes.starts_with(MAGIC_V3) { decoded.extend_from_slice(bytes); }
        else {
            let mut gz = GzDecoder::new(bytes);
            gz.read_to_end(&mut decoded).context("gzip decode failed; expected .splat4dpack v3")?;
        }
        let mut cur = Cursor::new(decoded.as_slice());
        let mut magic = [0_u8; 8];
        cur.read_exact(&mut magic)?;
        if &magic != MAGIC_V3 { bail!("not a splat4dpack v3 file"); }
        let version = cur.read_u32::<LittleEndian>()?;
        if version != FORMAT_VERSION { bail!("unsupported splat4dpack version {version}"); }

        let mut meta = None;
        let mut gaussians = Vec::new();
        let mut appearance = Vec::new();
        let mut aux = Vec::new();
        let mut keyframes = Vec::new();
        let mut mask_words = Vec::new();
        let mut temporal_mask_meta = None;
        let mut mlp_meta = None;
        let mut mlp_weights = Vec::new();
        let mut extra_chunks = Vec::new();
        while let Some(ch) = read_chunk(&mut cur)? {
            match &ch.tag {
                b"META" => meta = Some(serde_json::from_slice(&ch.payload)?),
                b"GAUS" => gaussians = bytemuck::cast_slice::<u8, GaussianRecordV3>(&ch.payload).to_vec(),
                b"APPR" => appearance = bytemuck::cast_slice::<u8, f32>(&ch.payload).to_vec(),
                b"AUX " => aux = bytemuck::cast_slice::<u8, f32>(&ch.payload).to_vec(),
                b"TMSK" => temporal_mask_meta = Some(serde_json::from_slice(&ch.payload)?),
                b"KEYF" => keyframes = bytemuck::cast_slice::<u8, f32>(&ch.payload).to_vec(),
                b"MASK" => mask_words = bytemuck::cast_slice::<u8, u32>(&ch.payload).to_vec(),
                b"MLPM" => mlp_meta = Some(serde_json::from_slice(&ch.payload)?),
                b"MLPW" => mlp_weights = bytemuck::cast_slice::<u8, f32>(&ch.payload).to_vec(),
                _ => extra_chunks.push(ch),
            }
        }
        let out = Self { meta: meta.context("missing META")?, gaussians, appearance, aux, keyframes, mask_words, temporal_mask_meta, mlp_meta, mlp_weights, extra_chunks };
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<()> {
        if self.gaussians.len() != self.meta.gaussian_count as usize {
            bail!("gaussian count mismatch: meta={} data={}", self.meta.gaussian_count, self.gaussians.len());
        }
        let rp = &self.meta.render_policy;
        if rp.allowed_render_types.is_empty() {
            bail!("render contract must declare at least one allowed render type");
        }
        if !rp.allowed_render_types.contains(&rp.required_render_type) {
            bail!("required render type is not listed as allowed");
        }
        if rp.default_render_type != rp.required_render_type || rp.reference_render_type != rp.required_render_type || rp.native_render_type != rp.required_render_type {
            bail!("render contract mismatch: native/default/reference/required must be identical; sorted and sort-free are not interchangeable");
        }
        for f in &rp.forbidden_render_types {
            if rp.allowed_render_types.contains(f) {
                bail!("render type cannot be both allowed and forbidden");
            }
        }
        if rp.required_render_type == RenderType::SortFreeMobileGs && (self.mlp_meta.is_none() || self.mlp_weights.is_empty()) {
            bail!("sort-free MobileGS pack requires MLPM/MLPW opacity+phi network weights");
        }
        if self.meta.has_sortfree_mlp && (self.mlp_meta.is_none() || self.mlp_weights.is_empty()) {
            bail!("pack declares sort-free MLP but MLPM/MLPW are missing");
        }
        if self.meta.has_temporal_masks {
            let t = self.temporal_mask_meta.as_ref().context("pack declares temporal masks but TMSK missing")?;
            if self.keyframes.len() != t.keyframe_count as usize { bail!("keyframe count mismatch"); }
            let need = t.keyframe_count as usize * t.words_per_mask as usize;
            if self.mask_words.len() < need { bail!("mask words truncated: need {}, have {}", need, self.mask_words.len()); }
        }
        Ok(())
    }

    pub fn chunk_payload(&self, tag: &[u8; 4]) -> Option<&[u8]> {
        self.extra_chunks.iter().find(|c| &c.tag == tag).map(|c| c.payload.as_slice())
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn normalize_quat(mut q: [f32; 4]) -> [f32; 4] {
    let n = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt().max(1e-20);
    for v in &mut q { *v /= n; }
    q
}

pub fn default_appearance_model_from_stride(stride: usize) -> AppearanceModel {
    if stride <= 3 { AppearanceModel::Rgb }
    else {
        let coeff_count = (stride / 3) as u32;
        let degree = ((coeff_count as f64).sqrt().round() as i32 - 1).max(0) as u8;
        AppearanceModel::Sh { degree, coeff_count, storage: TensorStorage::F32 }
    }
}
