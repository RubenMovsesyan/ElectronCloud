// ElectronCloudRenderer's CUDA kernels. build.cpp compiles this file to PTX
// (.build/cuda/electron_cloud_renderer.ptx, nvcc --ptx); the renderer JIT-loads it at
// runtime with cuModuleLoad, so nvcc never touches the C++ link.

#include <renderer/raymarch_params.h>

#include <common.h>

namespace {

__device__ float3 load3(const f32 v[3]) {
    return make_float3(v[0], v[1], v[2]);
}

__device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ float3 operator*(float3 a, f32 s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ f32 dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float3 normalize(float3 v) { return v * rsqrtf(dot(v, v)); }

// Where a ray meets a sphere: the [near, far] distances along `direction` (unit) between
// which it is inside. False if it misses.
__device__ bool raySphere(float3 origin, float3 direction, float3 center, f32 radius, f32& near, f32& far) {
    const float3 offset = origin - center;
    const f32 b = dot(offset, direction);
    const f32 c = dot(offset, offset) - radius * radius;
    const f32 discriminant = b * b - c;
    if (discriminant < 0.0f) {
        return false;
    }
    const f32 root = sqrtf(discriminant);
    near = -b - root;
    far = -b + root;
    return far > 0.0f;
}

// The electron cloud's density at world-space `position` and simulation `time` (atomic
// units). This is where |psi(r, t)|^2 of the atom's state will go.
//
// For now: a hard-coded test sphere at the origin, radius 1, density 1 at the centre falling
// linearly to 0 at the surface. `atom` and `time` are not used yet.
__device__ f32 checkRay(float3 position, const RayAtom& atom, f64 time) {
    (void)atom;
    (void)time;
    constexpr f32 kTestRadius = 1.0f;
    const f32 distance = sqrtf(dot(position, position));
    return fmaxf(0.0f, 1.0f - distance / kTestRadius);
}

__device__ u32 packRgba8(float3 color) {
    const u32 r = static_cast<u32>(__saturatef(color.x) * 255.0f + 0.5f);
    const u32 g = static_cast<u32>(__saturatef(color.y) * 255.0f + 0.5f);
    const u32 b = static_cast<u32>(__saturatef(color.z) * 255.0f + 0.5f);
    return r | (g << 8) | (b << 16) | (255u << 24);
}

} // namespace

// One thread per pixel. The ray is marched only where it is inside the atom's bounds (and
// within max_distance of the camera), `step_size` at a time, sampling the cloud's density and
// blending it front to back: each sample adds its colour, weighted by its own opacity and by
// how much light still gets through everything in front of it. The ray stops early once it
// is nearly opaque. Whatever light is left over shows the background.
extern "C" __global__ void raymarch(u32* pixels, RaymarchParams params) {
    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
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

    const RayAtom& atom = params.atom;
    const RayMarch& march = params.march;
    const RayAppearance& appearance = params.appearance;
    const float3 cloud_color = load3(appearance.cloud_color);
    const float3 background = load3(appearance.background_color);

    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);
    f32 accumulated_opacity = 0.0f;

    // Only the stretch of the ray inside the atom's bounds can see any density.
    f32 enter = 0.0f;
    f32 exit = 0.0f;
    if (raySphere(origin, direction, load3(atom.nucleus), atom.bounds_radius, enter, exit)) {
        const f32 start = fmaxf(enter, 0.0f); // the camera may be inside the bounds
        const f32 end = fminf(exit, march.max_distance);

        for (f32 distance = start + march.step_size; distance <= end; distance += march.step_size) {
            const f32 density = checkRay(origin + direction * distance, atom, params.time);
            if (density <= march.min_density) {
                continue;
            }

            // Beer-Lambert: how much of the light this step blocks.
            const f32 alpha = 1.0f - expf(-density * march.step_size * appearance.absorption);
            const float3 sample_color = cloud_color * density;

            // Front to back: only the light not already blocked in front of this sample counts.
            const f32 transmittance = 1.0f - accumulated_opacity;
            accumulated_color = accumulated_color + sample_color * (transmittance * alpha);
            accumulated_opacity += transmittance * alpha;

            if (accumulated_opacity >= march.opacity_cutoff) {
                accumulated_opacity = 1.0f;
                break;
            }
        }
    }

    pixels[y * params.width + x] = packRgba8(accumulated_color + background * (1.0f - accumulated_opacity));
}
