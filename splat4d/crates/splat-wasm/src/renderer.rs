use anyhow::{bail, Context, Result};
use splat_format::{PackedGaussian4D, ScenePackage, TemporalMaskHeader};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    eye_time: [f32; 4],
    viewport: [f32; 4],
    counts: [u32; 4],
    background: [f32; 4],
    quality: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    SortedReference,
    SortFree,
    Adaptive,
    Preview,
}

pub struct Renderer {
    pub mode: RenderMode,
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    camera: CameraUniform,
    scene: Option<SceneGpu>,
    bind_layout: wgpu::BindGroupLayout,
    condition_pipeline: wgpu::ComputePipeline,
    tile_pipeline: wgpu::ComputePipeline,
    sort_pipeline: wgpu::ComputePipeline,
    raster_sorted_pipeline: wgpu::ComputePipeline,
    raster_sortfree_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
}

struct SceneGpu {
    gaussian_count: u32,
    gaussians: wgpu::Buffer,
    appearance: wgpu::Buffer,
    conditioned: wgpu::Buffer,
    tile_lists: wgpu::Buffer,
    tile_counts: wgpu::Buffer,
    sort_keys: wgpu::Buffer,
    accum: wgpu::Buffer,
    camera: wgpu::Buffer,
    masks: Option<MaskGpu>,
    bind_group: wgpu::BindGroup,
}

struct MaskGpu {
    header: TemporalMaskHeader,
    keyframes: wgpu::Buffer,
    words: wgpu::Buffer,
}

impl Renderer {
    pub async fn new(canvas_id: &str, mode: RenderMode) -> std::result::Result<Self, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
        let document = window.document().ok_or_else(|| JsValue::from_str("no document"))?;
        let canvas = document.get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str("canvas not found"))?
            .dyn_into::<HtmlCanvasElement>()?;
        let width = canvas.client_width().max(1) as u32;
        let height = canvas.client_height().max(1) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| JsValue::from_str(&format!("create surface: {e}")))?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or_else(|| JsValue::from_str("WebGPU adapter not available"))?;
        let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("splat4d device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
        }, None).await.map_err(|e| JsValue::from_str(&format!("request device: {e}")))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bind layout"),
            entries: &[
                bgl(0, wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT, wgpu::BufferBindingType::Uniform),
                bgl(1, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl(2, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl(3, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl(4, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl(5, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl(6, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl(7, wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute layout"), bind_group_layouts: &[&bind_layout], push_constant_ranges: &[]
        });
        let condition_pipeline = compute(&device, &pl, "condition_4d", include_str!("../../../shaders/condition_4d.wgsl"), "main");
        let tile_pipeline = compute(&device, &pl, "tile_bin", include_str!("../../../shaders/tile_binning.wgsl"), "main");
        let sort_pipeline = compute(&device, &pl, "radix_sort_tiles", include_str!("../../../shaders/radix_sort_tiles.wgsl"), "main");
        let raster_sorted_pipeline = compute(&device, &pl, "raster_sorted", include_str!("../../../shaders/raster_sorted.wgsl"), "main");
        let raster_sortfree_pipeline = compute(&device, &pl, "raster_sortfree", include_str!("../../../shaders/raster_sortfree.wgsl"), "main");

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../../shaders/composite.wgsl").into()),
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("composite layout"), bind_group_layouts: &[&bind_layout], push_constant_ranges: &[] })),
            vertex: wgpu::VertexState { module: &composite_shader, entry_point: "vs", buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader, entry_point: "fs", compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None,
            multisample: wgpu::MultisampleState::default(), multiview: None,
        });

        let camera = CameraUniform {
            view: identity(), proj: identity(), view_proj: identity(), eye_time: [0.0,0.0,0.0,0.0],
            viewport: [width as f32, height as f32, 16.0, 16.0], counts: [0,0,0,0],
            background: [0.0,0.0,0.0,1.0], quality: [0,0,0,0],
        };

        Ok(Self { mode, canvas, surface, device, queue, config, camera, scene: None, bind_layout, condition_pipeline, tile_pipeline, sort_pipeline, raster_sorted_pipeline, raster_sortfree_pipeline, composite_pipeline })
    }

    pub fn upload_scene(&mut self, pkg: ScenePackage) -> Result<()> {
        let n = pkg.gaussians.len() as u32;
        if n == 0 { bail!("scene has no gaussians"); }
        self.camera.counts[0] = n;
        self.camera.background = pkg.meta.background;
        let pixels = self.config.width as u64 * self.config.height as u64;
        let tile_count = ((self.config.width + 15) / 16) * ((self.config.height + 15) / 16);
        let max_refs = (n as u64).saturating_mul(24).max(1);
        let gaussians = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gaussians"), contents: bytemuck::cast_slice(&pkg.gaussians), usage: wgpu::BufferUsages::STORAGE,
        });
        let appearance = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("appearance"), contents: bytemuck::cast_slice(&pkg.appearance), usage: wgpu::BufferUsages::STORAGE,
        });
        let conditioned = storage(&self.device, "conditioned", n as u64 * 128);
        let tile_lists = storage(&self.device, "tile lists", max_refs * 8);
        let tile_counts = storage(&self.device, "tile counts", tile_count as u64 * 4);
        let sort_keys = storage(&self.device, "sort keys", max_refs * 8);
        let accum = storage(&self.device, "accum rgba+alpha", pixels * 32);
        let camera = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"), contents: bytemuck::bytes_of(&self.camera), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let masks = match pkg.mask_header {
            Some(header) => Some(MaskGpu {
                header,
                keyframes: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("mask keyframes"), contents: bytemuck::cast_slice(&pkg.keyframes), usage: wgpu::BufferUsages::STORAGE }),
                words: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("mask words"), contents: bytemuck::cast_slice(&pkg.mask_words), usage: wgpu::BufferUsages::STORAGE }),
            }),
            None => None,
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bind group"), layout: &self.bind_layout,
            entries: &[
                bg(0, camera.as_entire_binding()), bg(1, gaussians.as_entire_binding()), bg(2, appearance.as_entire_binding()),
                bg(3, conditioned.as_entire_binding()), bg(4, tile_lists.as_entire_binding()), bg(5, tile_counts.as_entire_binding()),
                bg(6, sort_keys.as_entire_binding()), bg(7, accum.as_entire_binding()),
            ],
        });
        self.scene = Some(SceneGpu { gaussian_count: n, gaussians, appearance, conditioned, tile_lists, tile_counts, sort_keys, accum, camera, masks, bind_group });
        Ok(())
    }

    pub fn set_camera(&mut self, view: &[f32], proj: &[f32], eye: [f32;3], timestamp: f32) {
        self.camera.view = mat(view);
        self.camera.proj = mat(proj);
        self.camera.view_proj = mul4(self.camera.proj, self.camera.view);
        self.camera.eye_time = [eye[0], eye[1], eye[2], timestamp];
        self.camera.viewport = [self.config.width as f32, self.config.height as f32, 16.0, 16.0];
    }

    pub fn render(&mut self) -> Result<()> {
        if self.canvas.width() != self.config.width || self.canvas.height() != self.config.height {
            self.config.width = self.canvas.width().max(1);
            self.config.height = self.canvas.height().max(1);
            self.surface.configure(&self.device, &self.config);
        }
        let scene = self.scene.as_ref().context("no scene loaded")?;
        self.camera.quality[0] = match self.mode { RenderMode::SortedReference => 0, RenderMode::SortFree => 1, RenderMode::Adaptive => 2, RenderMode::Preview => 3 };
        self.queue.write_buffer(&scene.camera, 0, bytemuck::bytes_of(&self.camera));
        let frame = self.surface.get_current_texture().context("surface frame")?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("splat render encoder") });
        {
            let mut c = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("condition/cull/bin/sort/raster"), timestamp_writes: None });
            c.set_bind_group(0, &scene.bind_group, &[]);
            c.set_pipeline(&self.condition_pipeline);
            c.dispatch_workgroups((scene.gaussian_count + 127) / 128, 1, 1);
            c.set_pipeline(&self.tile_pipeline);
            c.dispatch_workgroups((scene.gaussian_count + 127) / 128, 1, 1);
            match self.mode {
                RenderMode::SortedReference | RenderMode::Adaptive => {
                    c.set_pipeline(&self.sort_pipeline);
                    c.dispatch_workgroups((((self.config.width + 15)/16) * ((self.config.height + 15)/16)).max(1), 1, 1);
                    c.set_pipeline(&self.raster_sorted_pipeline);
                }
                RenderMode::SortFree | RenderMode::Preview => c.set_pipeline(&self.raster_sortfree_pipeline),
            }
            c.dispatch_workgroups((self.config.width + 7) / 8, (self.config.height + 7) / 8, 1);
        }
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite"), color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None,
            });
            rp.set_pipeline(&self.composite_pipeline);
            rp.set_bind_group(0, &scene.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        self.queue.submit(Some(enc.finish()));
        frame.present();
        Ok(())
    }
}

fn bgl(binding: u32, stages: wgpu::ShaderStages, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: stages, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn bg<'a>(binding: u32, resource: wgpu::BindingResource<'a>) -> wgpu::BindGroupEntry<'a> { wgpu::BindGroupEntry { binding, resource } }
fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
}
fn compute(device: &wgpu::Device, layout: &wgpu::PipelineLayout, label: &str, src: &str, entry: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(layout), module: &module, entry_point: entry, compilation_options: Default::default(), cache: None })
}
fn identity() -> [[f32;4];4] { [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]] }
fn mat(v: &[f32]) -> [[f32;4];4] { [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[8],v[9],v[10],v[11]],[v[12],v[13],v[14],v[15]]] }
fn mul4(a: [[f32;4];4], b: [[f32;4];4]) -> [[f32;4];4] { let mut r = [[0.;4];4]; for i in 0..4 { for j in 0..4 { for k in 0..4 { r[i][j] += a[i][k]*b[k][j]; } } } r }
