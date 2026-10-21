// ElectronCloudRenderer's CUDA kernels. build.cpp compiles this file to PTX
// (.build/cuda/electron_cloud_renderer.ptx, nvcc --ptx); the renderer JIT-loads it at
// runtime with cuModuleLoad, so nvcc never touches the C++ link.

#include <renderer/raymarch_params.h>

#include <common.h>

namespace {

// How strongly density absorbs light per unit length (Beer-Lambert): a step of length `step`
// through density d is 1 - exp(-d * step * kAbsorption) opaque.
constexpr f32 kAbsorption = 10.0f;
// Samples below this density are treated as empty space.
constexpr f32 kMinDensity = 0.001f;
// Stop marching once the ray is this opaque; nothing behind it would show.
constexpr f32 kOpaque = 0.95f;

__device__ float3 load3(const f32 v[3]) {
    return make_float3(v[0], v[1], v[2]);
}

__device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 operator*(float3 a, f32 s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ f32 dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 normalize(float3 v) { return v * rsqrtf(dot(v, v)); }

// The cloud's density at `position`. For now, a test sphere: 1 at the centre falling
// linearly to 0 at the surface, and 0 outside.
__device__ f32 checkRay(float3 position, const RaySphere& sphere) {
    const float3 offset = position - load3(sphere.center);
    const f32 distance = sqrtf(dot(offset, offset));
    return fmaxf(0.0f, 1.0f - distance / sphere.radius);
}

__device__ unsigned packRgba8(f32 r, f32 g, f32 b) {
    const unsigned R = static_cast<unsigned>(__saturatef(r) * 255.0f + 0.5f);
    const unsigned G = static_cast<unsigned>(__saturatef(g) * 255.0f + 0.5f);
    const unsigned B = static_cast<unsigned>(__saturatef(b) * 255.0f + 0.5f);
    return R | (G << 8) | (B << 16) | (255u << 24);
}

} // namespace

// One thread per pixel. The ray starts at the camera and advances `step` at a time out to
// `max_distance`, sampling the cloud's density at each step and blending it front to back:
// each sample adds its colour, weighted by its own opacity and by how much light still gets
// through everything in front of it. The ray stops early once it is nearly opaque. Whatever
// light is left over shows the background.
extern "C" __global__ void raymarch(unsigned* pixels, RaymarchParams params) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    // Pixel centre -> [-1, 1] screen coordinates, +y up (pixel row 0 is the top).
    const f32 sx = ((x + 0.5f) / params.width) * 2.0f - 1.0f;
    const f32 sy = 1.0f - ((y + 0.5f) / params.height) * 2.0f;

    const RayCamera& camera = params.camera;
    const float3 origin = load3(camera.position);
    const float3 direction = normalize(load3(camera.forward) + load3(camera.right) * (sx * camera.tan_half_fov_y * camera.aspect) +
                                       load3(camera.up) * (sy * camera.tan_half_fov_y));

    const float3 cloud_color = make_float3(0.3f, 0.6f, 1.0f);
    const float3 background = make_float3(0.02f, 0.02f, 0.03f);

    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    f32 accumulated_opacity = 0.0f;

    const unsigned steps = static_cast<unsigned>(params.max_distance / params.step + 0.5f);
    for (unsigned i = 1; i <= steps; i++) {
        const f32 t = params.step * static_cast<f32>(i);
        const f32 density = checkRay(origin + direction * t, params.sphere);
        if (density <= kMinDensity) {
            continue;
        }

        // Beer-Lambert: how much of the light this step blocks.
        const f32 alpha = 1.0f - expf(-density * params.step * kAbsorption);
        const float3 sample_color = cloud_color * density;

        // Front to back: only the light not already blocked in front of this sample counts.
        const f32 transmittance = 1.0f - accumulated_opacity;
        accumulated_color = accumulated_color + sample_color * (transmittance * alpha);
        accumulated_opacity += transmittance * alpha;

        if (accumulated_opacity >= kOpaque) {
            accumulated_opacity = 1.0f;
            break;
        }
    }

    const float3 color = accumulated_color + background * (1.0f - accumulated_opacity);
    pixels[y * params.width + x] = packRgba8(color.x, color.y, color.z);
}
