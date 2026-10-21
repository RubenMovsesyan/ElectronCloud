// ElectronCloudRenderer's shaders, in sokol-shdc's annotated GLSL. build.cpp generates
// .build/shaders/electron_cloud.glsl.h from this file.
//
// CUDA can't write a Vulkan image in this engine, only a buffer, so the CUDA kernels write
// RGBA8 pixels into a storage buffer and this pass copies them into the renderer's output
// image, one fragment per pixel.

// Fullscreen triangle from the vertex index alone (same as GameEngine's fullscreen_vs). v_uv
// is a top-left-origin screen uv; see the engine's engine.glsl for the Vulkan y-flip note.
@vs ec_fullscreen_vs
out vec2 v_uv;
void main() {
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_uv.x * 2.0 - 1.0, 1.0 - v_uv.y * 2.0, 0.0, 1.0);
}
@end

@fs ec_pixels_fs
// Written by CUDA: pixel (x, y), y = 0 at the top, at index y * width + x, packed RGBA8
// with red in the lowest byte.
struct ec_pixel {
    uint rgba;
};

layout(binding=0) readonly buffer ec_pixels {
    ec_pixel pixels[];
};

layout(binding=0) uniform ec_pixels_params {
    vec4 size; // x = width, y = height, in pixels
};

in vec2 v_uv;
out vec4 out_color;

void main() {
    const ivec2 pixel = ivec2(gl_FragCoord.xy);
    out_color = unpackUnorm4x8(pixels[pixel.y * int(size.x) + pixel.x].rgba);
}
@end

@program ec_pixels ec_fullscreen_vs ec_pixels_fs
