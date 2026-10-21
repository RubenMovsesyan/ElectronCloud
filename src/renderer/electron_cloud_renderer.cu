// ElectronCloudRenderer's CUDA kernels. build.cpp compiles this file to PTX
// (.build/cuda/electron_cloud_renderer.ptx, nvcc --ptx); the renderer JIT-loads it at
// runtime with cuModuleLoad, so nvcc never touches the C++ link.

#include <renderer/raymarch_params.h>

namespace {

__device__ float3 load3(const float v[3]) {
    return make_float3(v[0], v[1], v[2]);
}

__device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 normalize(float3 v) { return v * rsqrtf(dot(v, v)); }

// Whether a ray at `position` has intersected. For now: inside (or on) the sphere.
__device__ bool checkRay(float3 position, const RaySphere& sphere) {
    const float3 offset = position - load3(sphere.center);
    return dot(offset, offset) <= sphere.radius * sphere.radius;
}

__device__ unsigned packRgba8(float r, float g, float b) {
    const unsigned R = static_cast<unsigned>(__saturatef(r) * 255.0f + 0.5f);
    const unsigned G = static_cast<unsigned>(__saturatef(g) * 255.0f + 0.5f);
    const unsigned B = static_cast<unsigned>(__saturatef(b) * 255.0f + 0.5f);
    return R | (G << 8) | (B << 16) | (255u << 24);
}

} // namespace

// One thread per pixel. The ray starts at the camera and advances `step` at a time until
// checkRay() says it intersected or it is `max_distance` from the camera. A hit is shaded by
// how far the ray travelled (nearer = brighter) so the shape reads; a miss is the background.
extern "C" __global__ void raymarch(unsigned* pixels, RaymarchParams params) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    // Pixel centre -> [-1, 1] screen coordinates, +y up (pixel row 0 is the top).
    const float sx = ((x + 0.5f) / params.width) * 2.0f - 1.0f;
    const float sy = 1.0f - ((y + 0.5f) / params.height) * 2.0f;

    const RayCamera& camera = params.camera;
    const float3 origin = load3(camera.position);
    const float3 direction = normalize(load3(camera.forward) + load3(camera.right) * (sx * camera.tan_half_fov_y * camera.aspect) +
                                       load3(camera.up) * (sy * camera.tan_half_fov_y));

    const unsigned steps = static_cast<unsigned>(params.max_distance / params.step + 0.5f);
    unsigned color = packRgba8(0.02f, 0.02f, 0.03f);
    for (unsigned i = 1; i <= steps; i++) {
        const float t = params.step * static_cast<float>(i);
        if (checkRay(origin + direction * t, params.sphere)) {
            const float brightness = 1.0f - t / params.max_distance;
            color = packRgba8(brightness, brightness * 0.9f, brightness * 0.6f);
            break;
        }
    }

    pixels[y * params.width + x] = color;
}
