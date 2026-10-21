// ElectronCloudRenderer's CUDA kernels. build.cpp compiles this file to PTX with clang's
// CUDA support (.build/cuda/electron_cloud_renderer.ptx); the renderer JIT-loads it at
// runtime with cuModuleLoad, so device code never touches the C++ link.

#include <renderer/raymarch_params.h>

#include <common.h>

#include <Eigen/Core>

using Eigen::Vector3f;

namespace renderer {

namespace {

// Where a ray meets a sphere: the [near, far] distances along `direction` (unit) between
// which it is inside. False if it misses.
__device__ bool raySphere(const Vector3f& origin, const Vector3f& direction, const Vector3f& center, f32 radius, f32& near, f32& far) {
    const Vector3f offset = origin - center;
    const f32 b = offset.dot(direction);
    const f32 c = offset.squaredNorm() - radius * radius;
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
__device__ f32 checkRay(const Vector3f& position, const Atom& atom, f64 time) {
    (void)atom;
    (void)time;
    constexpr f32 kTestRadius = 1.0f;
    return fmaxf(0.0f, 1.0f - position.norm() / kTestRadius);
}

__device__ u32 packRgba8(const Vector3f& color) {
    const u32 r = static_cast<u32>(__saturatef(color.x()) * 255.0f + 0.5f);
    const u32 g = static_cast<u32>(__saturatef(color.y()) * 255.0f + 0.5f);
    const u32 b = static_cast<u32>(__saturatef(color.z()) * 255.0f + 0.5f);
    return r | (g << 8) | (b << 16) | (255u << 24);
}

} // namespace

} // namespace renderer

// One thread per pixel. The ray is marched only where it is inside the atom's bounds (and
// within max_distance of the camera), `step_size` at a time, sampling the cloud's density and
// blending it front to back: each sample adds its colour, weighted by its own opacity and by
// how much light still gets through everything in front of it. The ray stops early once it
// is nearly opaque. Whatever light is left over shows the background.
//
// extern "C" and outside the namespace so the PTX entry point is plainly named "raymarch".
extern "C" __global__ void raymarch(u32* pixels, renderer::RaymarchParams params) {
    using namespace renderer;

    const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    // Pixel centre -> [-1, 1] screen coordinates, +y up (pixel row 0 is the top).
    const f32 sx = ((x + 0.5f) / params.width) * 2.0f - 1.0f;
    const f32 sy = 1.0f - ((y + 0.5f) / params.height) * 2.0f;

    const RayCamera& camera = params.camera;
    const Vector3f direction =
        (camera.forward + camera.right * (sx * camera.tan_half_fov_y * camera.aspect) + camera.up * (sy * camera.tan_half_fov_y)).normalized();

    const Atom& atom = params.atom;
    const RaymarchSettings& march = params.march;
    const Appearance& appearance = params.appearance;

    Vector3f accumulated_color = Vector3f::Zero();
    f32 accumulated_opacity = 0.0f;

    // Only the stretch of the ray inside the atom's bounds can see any density.
    f32 enter = 0.0f;
    f32 exit = 0.0f;
    if (raySphere(camera.position, direction, atom.nucleus, atom.bounds_radius, enter, exit)) {
        const f32 start = fmaxf(enter, 0.0f); // the camera may be inside the bounds
        const f32 end = fminf(exit, march.max_distance);

        for (f32 distance = start + march.step_size; distance <= end; distance += march.step_size) {
            const f32 density = checkRay(camera.position + direction * distance, atom, params.time);
            if (density <= march.min_density) {
                continue;
            }

            // Beer-Lambert: how much of the light this step blocks.
            const f32 alpha = 1.0f - expf(-density * march.step_size * appearance.absorption);

            // Front to back: only the light not already blocked in front of this sample counts.
            const f32 transmittance = 1.0f - accumulated_opacity;
            accumulated_color += appearance.cloud_color * (density * transmittance * alpha);
            accumulated_opacity += transmittance * alpha;

            if (accumulated_opacity >= march.opacity_cutoff) {
                accumulated_opacity = 1.0f;
                break;
            }
        }
    }

    pixels[y * params.width + x] = packRgba8(accumulated_color + appearance.background_color * (1.0f - accumulated_opacity));
}
