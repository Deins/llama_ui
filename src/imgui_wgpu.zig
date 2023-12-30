const std = @import("std");
const builtin = @import("builtin");
const imgui = @import("imgui");
const gpu = @import("core").gpu;
//pub const tracy = @import("ztracy");

const Vert = imgui.DrawVert;
const Indice = imgui.DrawIdx;

var is_init = false;
var pipeline: *gpu.RenderPipeline = undefined;
var font_texture: *gpu.Texture = undefined;
var font_texture_view: *gpu.TextureView = undefined;
var uniform_buffer: *gpu.Buffer = undefined;
var uniform_bind_group: *gpu.BindGroup = undefined;
var sampler: *gpu.Sampler = undefined;
var buffers: [2]?*gpu.Buffer = .{ null, null };

const DrawUniforms = extern struct {
    mvp: [4][4]f32 = .{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    },
};

pub const GpuContext = struct {
    device: *gpu.Device,
    queue: *gpu.Queue,
    swap_chain_format: gpu.Texture.Format,
};

pub const FrameContext = struct {
    dt: f32, // delta time
    device: *gpu.Device,
    queue: *gpu.Queue,
    swap_chain: ?*gpu.SwapChain,
};

pub fn init(gpu_ctx: GpuContext) void {
    // const t_imgui_init = tracy.ZoneN(@src(), "init imgui_wgpu");
    // defer t_imgui_init.End();
    if (is_init) unreachable;
    const device = gpu_ctx.device;
    const queue = gpu_ctx.queue;

    const io = imgui.GetIO();
    io.BackendFlags.RendererHasVtxOffset = true;
    {
        // const t_tex = tracy.ZoneN(@src(), "create font");
        // defer t_tex.End();
        var pixels: ?[*]u8 = undefined;
        var width: i32 = 0;
        var height: i32 = 0;
        const bytes_per_channel: u32 = 4;
        {
            // const t_tex_bake = tracy.ZoneN(@src(), "bake");
            // defer t_tex_bake.End();
            io.Fonts.?.GetTexDataAsRGBA32(&pixels, &width, &height);
        }
        const img_size = gpu.Extent3D{ .width = @as(u32, @intCast(width)), .height = @as(u32, @intCast(height)) };
        const tex_desc = gpu.Texture.Descriptor{
            .label = "font_texture",
            .format = .rgba8_unorm,
            .size = img_size,
            .usage = .{
                .texture_binding = true,
                .copy_dst = true,
                .render_attachment = true,
            },
        };
        font_texture = device.createTexture(&tex_desc);
        const data_layout = gpu.Texture.DataLayout{
            .bytes_per_row = @as(u32, @intCast(width)) * bytes_per_channel,
            .rows_per_image = @as(u32, @intCast(height)),
        };
        {
            // const t_tex_upload = tracy.ZoneN(@src(), "upload");
            // defer t_tex_upload.End();
            queue.writeTexture(&.{ .texture = font_texture }, &data_layout, &img_size, pixels.?[0..@as(usize, @intCast(width * height * bytes_per_channel))]);
            font_texture_view = font_texture.createView(&gpu.TextureView.Descriptor{});
        }
        io.Fonts.?.SetTexID(font_texture_view);
    }

    //const t_create_shader_module = tracy.ZoneN(@src(), "compile shader");
    const shader_module = device.createShaderModuleWGSL("imgui", shader_src);
    //t_create_shader_module.End();
    defer shader_module.release();

    const blend = gpu.BlendState{
        .color = .{
            .operation = .add,
            .src_factor = .src_alpha,
            .dst_factor = .one_minus_src_alpha,
        },
        .alpha = .{},
    };
    const col_target = gpu.ColorTargetState{
        .format = gpu_ctx.swap_chain_format,
        .blend = &blend,
        .write_mask = gpu.ColorWriteMaskFlags.all,
    };
    const frag_state = gpu.FragmentState.init(.{
        .module = shader_module,
        .entry_point = "frag_main",
        .targets = &.{col_target},
    });
    const vertex_attributes = [_]gpu.VertexAttribute{
        .{ .format = .float32x2, .offset = @offsetOf(Vert, "pos"), .shader_location = 0 },
        .{ .format = .float32x2, .offset = @offsetOf(Vert, "uv"), .shader_location = 1 },
        .{ .format = .unorm8x4, .offset = @offsetOf(Vert, "col"), .shader_location = 2 },
    };
    const vertex_buffer_layout = gpu.VertexBufferLayout.init(.{
        .array_stride = @sizeOf(Vert),
        .step_mode = .vertex,
        .attributes = &vertex_attributes,
    });
    // const bind_group_layouts : []*const gpu.BindGroupLayout = .{
    // };
    const pipeline_desc = gpu.RenderPipeline.Descriptor{
        .label = "imgui",
        .fragment = &frag_state,
        //.layout = device.createPipelineLayout(&gpu.PipelineLayout.Descriptor.init(.{})),
        //.layout = gpu.PipelineLayout.init(.{ .bind_group_layouts = .{}}),
        .depth_stencil = null,
        .vertex = gpu.VertexState.init(.{
            .module = shader_module,
            .entry_point = "vert_main",
            .buffers = &.{vertex_buffer_layout},
        }),
        .multisample = .{},
        .primitive = .{},
    };
    //const t_cpp = tracy.ZoneN(@src(), "render pipeline");
    pipeline = device.createRenderPipeline(&pipeline_desc);
    //t_cpp.End();

    {
        // const t_unif = tracy.ZoneN(@src(), "gpu: bindings & uniforms");
        // defer t_unif.End();

        const uniform_desc = gpu.Buffer.Descriptor{
            .label = "imgui",
            .usage = .{ .copy_dst = true, .uniform = true },
            .size = @sizeOf(DrawUniforms),
        };
        uniform_buffer = device.createBuffer(&uniform_desc);

        sampler = device.createSampler(&.{
            .mag_filter = .linear,
            .min_filter = .linear,
        });

        uniform_bind_group = device.createBindGroup(
            &gpu.BindGroup.Descriptor.init(.{
                .layout = pipeline.getBindGroupLayout(0),
                .entries = &.{
                    gpu.BindGroup.Entry.buffer(0, uniform_buffer, 0, @sizeOf(DrawUniforms)),
                    gpu.BindGroup.Entry.sampler(1, sampler),
                    gpu.BindGroup.Entry.textureView(2, font_texture_view),
                },
            }),
        );
    }

    is_init = true;
}

pub fn renderDrawData(draw_data: *imgui.DrawData, ctx: FrameContext, pass_encoder: *gpu.CommandEncoder, pass: *gpu.RenderPassEncoder) void {
    if (!is_init) unreachable;
    const device = ctx.device;
    const queue = ctx.queue;

    if (draw_data.DisplaySize.x <= 0.0 or draw_data.DisplaySize.y <= 0.0)
        return;

    const vert_count = draw_data.TotalVtxCount;
    const indice_count = draw_data.TotalIdxCount;

    if (vert_count <= 0 or indice_count <= 0) return;

    if (buffers[0]) |b| {
        b.release();
        buffers[0] = null;
    }
    if (buffers[1]) |b| {
        b.release();
        buffers[1] = null;
    }

    // create vbo & ibo
    // TODO: buffer reuse & bind groups?
    const vbo = device.createBuffer(&.{ .usage = .{ .copy_dst = true, .vertex = true }, .size = std.mem.alignForward(@as(usize, @intCast(vert_count * @sizeOf(Vert))), 4), .mapped_at_creation = true });
    buffers[0] = vbo; //defer vbo.destroy();
    const vb = vbo.getMappedRange(Vert, 0, @as(usize, @intCast(vert_count))).?;
    var vtx_dst = vb;

    const ibo = device.createBuffer(&.{ .usage = .{ .copy_dst = true, .index = true }, .size = std.mem.alignForward(@as(usize, @intCast(indice_count * @sizeOf(Indice))), 4), .mapped_at_creation = true });
    buffers[1] = ibo; //defer ibo.destroy();
    const ib = ibo.getMappedRange(Indice, 0, @as(usize, @intCast(indice_count))).?;
    var idx_dst = ib;

    var i: u32 = 0;
    const cmd_lists = draw_data.CmdLists.?;
    while (i < draw_data.CmdListsCount) : (i += 1) {
        const cmd = cmd_lists[i];
        std.mem.copy(Vert, vtx_dst, cmd.VtxBuffer.Data.?[0..cmd.VtxBuffer.Size]);
        std.mem.copy(Indice, idx_dst, cmd.IdxBuffer.Data.?[0..cmd.IdxBuffer.Size]);
        vtx_dst = vtx_dst[cmd.VtxBuffer.Size..];
        idx_dst = idx_dst[cmd.IdxBuffer.Size..];
    }
    vbo.unmap();
    ibo.unmap();

    // Setup orthographic projection matrix into our constant buffer
    // Our visible imgui space lies from draw_data->DisplayPos (top left) to draw_data->DisplayPos+data_data->DisplaySize (bottom right).
    var uniforms: DrawUniforms = .{};
    {
        const L: f32 = draw_data.DisplayPos.x;
        const R: f32 = draw_data.DisplayPos.x + draw_data.DisplaySize.x;
        const T: f32 = draw_data.DisplayPos.y;
        const B: f32 = draw_data.DisplayPos.y + draw_data.DisplaySize.y;
        uniforms.mvp =
            .{
            .{ 2.0 / (R - L), 0.0, 0.0, 0.0 },
            .{ 0.0, 2.0 / (T - B), 0.0, 0.0 },
            .{ 0.0, 0.0, 0.5, 0.0 },
            .{ (R + L) / (L - R), (T + B) / (B - T), 0.5, 1.0 },
        };
        queue.writeBuffer(uniform_buffer, 0, &[_]DrawUniforms{uniforms});
    }

    const blend_color: gpu.Color = .{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 };
    { // setup render state
        pass.setViewport(0, 0, draw_data.FramebufferScale.x * draw_data.DisplaySize.x, draw_data.FramebufferScale.y * draw_data.DisplaySize.y, 0, 1.0);
        pass.setPipeline(pipeline);
        pass.setVertexBuffer(0, vbo, 0, gpu.whole_size);
        pass.setIndexBuffer(ibo, .uint16, 0, gpu.whole_size);
        pass.setBindGroup(0, uniform_bind_group, null);
        pass.setBlendConstant(&blend_color);
    }

    { // render
        var global_vtx_offset: u32 = 0;
        var global_idx_offset: u32 = 0;
        const clip_scale = draw_data.FramebufferScale;
        const clip_off = draw_data.DisplayPos;
        var n: i32 = 0;
        while (n < draw_data.CmdListsCount) : (n += 1) {
            const cmd_list = cmd_lists[@as(usize, @intCast(n))];
            var cmd_i: i32 = 0;
            while (cmd_i < cmd_list.CmdBuffer.Size) : (cmd_i += 1) {
                const pcmd = &cmd_list.CmdBuffer.Data.?[@as(usize, @intCast(cmd_i))];
                if (pcmd.UserCallback != null) {
                    // User callback, registered via ImDrawList::AddCallback()
                    // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                    if (pcmd.UserCallback == imgui.DrawCallback_ResetRenderState) { // setup render state
                        pass.setViewport(0, 0, draw_data.FramebufferScale.x * draw_data.DisplaySize.x, draw_data.FramebufferScale.y * draw_data.DisplaySize.y, 0, 1.0);
                        pass.setPipeline(pipeline);
                        pass.setVertexBuffer(0, vbo, 0, gpu.whole_size);
                        pass.setIndexBuffer(ibo, .uint16, 0, gpu.whole_size);
                        pass.setBindGroup(0, uniform_bind_group, null);
                        pass.setBlendConstant(&blend_color);
                    } else if (pcmd.UserCallback) |cb| cb(cmd_list, pcmd);
                } else {
                    // Bind custom texture
                    const tex_id = pcmd.GetTexID();
                    if (tex_id != null and @intFromPtr(tex_id.?) != @intFromPtr(font_texture_view))
                        // FIXME: TODO: more than 1 texture, requires bind group for each, currently just draws everything with default texture bind group!
                        std.log.warn("imgui multiple textures not implemented! {*}", .{tex_id});

                    // Project scissor/clipping rectangles into framebuffer space
                    const clip_min: imgui.Vec2 = .{ .x = (pcmd.ClipRect.x - clip_off.x) * clip_scale.x, .y = (pcmd.ClipRect.y - clip_off.y) * clip_scale.y };
                    const clip_max: imgui.Vec2 = .{ .x = (pcmd.ClipRect.z - clip_off.x) * clip_scale.x, .y = (pcmd.ClipRect.w - clip_off.y) * clip_scale.y };
                    if (clip_max.x <= clip_min.x or clip_max.y <= clip_min.y)
                        continue;

                    // Apply scissor/clipping rectangle, Draw
                    pass.setScissorRect(@as(u32, @intFromFloat(clip_min.x)), @as(u32, @intFromFloat(clip_min.y)), @as(u32, @intFromFloat(clip_max.x - clip_min.x)), @as(u32, @intFromFloat(clip_max.y - clip_min.y)));
                    pass.drawIndexed(pcmd.ElemCount, 1, pcmd.IdxOffset + global_idx_offset, @as(i32, @intCast(pcmd.VtxOffset + global_vtx_offset)), 0);
                }
            }
            global_idx_offset += cmd_list.IdxBuffer.Size;
            global_vtx_offset += cmd_list.VtxBuffer.Size;
        }
    }
    _ = pass_encoder;
}

pub fn deinit() void {
    if (!is_init) unreachable;
    is_init = false;
    pipeline.release();
    font_texture_view.release();
    font_texture.release();
    uniform_buffer.release();
    uniform_bind_group.release();
    sampler.release();
}

const shader_src =
    \\struct VertexOutput {
    \\    @builtin(position) pos: vec4<f32>,
    \\    @location(0) uv: vec2<f32>,
    \\    @location(1) col: vec4<f32>,
    \\};
    \\
    \\struct DrawUniforms {
    \\    mvp : mat4x4<f32>,
    \\};
    \\@group(0) @binding(0) var<uniform> draw_uniforms: DrawUniforms;
    \\
    \\@vertex
    \\fn vert_main(
    \\    @location(0) pos: vec4<f32>,
    \\    @location(1) uv: vec2<f32>,
    \\    @location(2) col: vec4<f32>,
    \\) -> VertexOutput {
    \\    var out: VertexOutput;
    \\    out.uv = uv;
    \\    out.col = col;
    \\    out.pos = draw_uniforms.mvp * pos;
    \\    return out;
    \\}
    \\
    \\@group(0) @binding(1) var tex_sampler: sampler;
    \\@group(0) @binding(2) var texture: texture_2d<f32>;
    \\
    \\@fragment
    \\fn frag_main(in: VertexOutput) -> @location(0) vec4<f32> {
    \\  return in.col * textureSample(texture, tex_sampler, in.uv);
    \\}
;
