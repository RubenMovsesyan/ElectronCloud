#pragma once

#ifndef GAMEENGINE_CUDA
#error "ElectronCloudRenderer needs CUDA: build.cpp defines GAMEENGINE_CUDA when it finds the toolkit"
#endif

#include <renderer/raymarch_params.h>

#include <gpu/cross_semaphore.hpp>
#include <gpu/cuda_interop.hpp>
#include <gpu/external_buffer.hpp>
#include <gpu/handle.hpp>
#include <graphics/primitives/pass/fullscreen_pass.hpp>
#include <graphics/primitives/pass/render_target.hpp>
#include <graphics/primitives/scene/camera.hpp>
#include <graphics/renderer.hpp>

#include <common.h>

#include <Eigen/Dense>

#include <cuda.h>

#include <array>
#include <optional>

namespace renderer {

// The C++-side description of what to draw and how. Each mirrors a plain-C struct in
// raymarch_params.h (Ray*), which is what the kernel actually receives; see that file for
// units.

// One hydrogen-like eigenstate |n l m>, weighted by amplitude * e^(i * phase).
struct QuantumState {
        i32 n = 1; // principal, >= 1
        i32 l = 0; // azimuthal, 0 .. n-1
        i32 m = 0; // magnetic, -l .. l
        f32 amplitude = 1.0f;
        f32 phase = 0.0f; // radians, at time 0
};

// A nucleus and the electron's state around it -- everything the cloud's density depends
// on. A transition between energy levels is a superposition of its start and end states
// (state_count = 2) whose amplitudes the app moves over time.
struct Atom {
        Eigen::Vector3f nucleus = Eigen::Vector3f::Zero(); // world space
        f32 nuclear_charge = 1.0f;                          // Z: 1 = hydrogen
        f32 bohr_radius = 0.1f;                             // world units per Bohr radius
        f32 bounds_radius = 1.5f;                           // world units; density is 0 outside
        std::array<QuantumState, 2> states{};
        u32 state_count = 1; // 1 or 2
};

// How rays are marched: quality against speed.
struct RaymarchSettings {
        f32 step_size = 0.1f;       // distance advanced per sample
        f32 max_distance = 5.0f;    // a ray gives up this far from the camera
        f32 min_density = 0.001f;   // samples at or below this are empty space
        f32 opacity_cutoff = 0.95f; // a ray stops once this opaque
};

// Purely cosmetic: how density turns into colour.
struct Appearance {
        Eigen::Vector3f cloud_color{0.3f, 0.6f, 1.0f};        // colour of density 1, scaled by density
        Eigen::Vector3f background_color{0.02f, 0.02f, 0.03f}; // shows through where the cloud isn't opaque
        f32 absorption = 10.0f;                                // Beer-Lambert strength
};

// ElectronCloud's own implementation of GameEngine's graphics::Renderer concept. Lives here,
// not in the engine: it plugs into graphics::Presenter like any engine renderer, but it is
// specific to this project.
//
// A CUDA ray marcher. Every frame a CUDA kernel marches one ray per pixel from the camera and
// writes RGBA8 pixels into a buffer shared with Vulkan (CUDA can't write a Vulkan image in
// this engine, only a buffer); a fullscreen pass then copies the buffer into the output image.
// CUDA runs here, on the render thread -- no worker.
//
// Per frame, two timeline semaphores keep CUDA and Vulkan off the pixel buffer at the same
// time: CUDA waits for `render_done` (the previous frame's copy has finished reading), writes,
// and signals `cuda_done`; the graphics queue waits for `cuda_done` before the copy.
//
// sokol_gfx is single-threaded: construct, call and destroy on the Application's thread.
class ElectronCloudRenderer {
    public:
        explicit ElectronCloudRenderer(RaymarchSettings settings = {}, Appearance appearance = {});
        ~ElectronCloudRenderer();

        ElectronCloudRenderer(const ElectronCloudRenderer&) = delete;
        ElectronCloudRenderer& operator=(const ElectronCloudRenderer&) = delete;

        void setSettings(const RaymarchSettings& settings);
        void setAppearance(const Appearance& appearance);

        // Where the rays start and which way they point, and the simulation time (atomic units
        // of time, hbar / E_h ~= 2.419e-17 s) the cloud is drawn at. Call once per frame,
        // before render().
        void beginFrame(const graphics::primitives::scene::Camera& camera, f64 time);

        // The frame's scene: the atom whose electron cloud the rays march through.
        void submit(const Atom& atom);

        // Marches the rays at width x height pixels and returns a view of the image. Does not
        // call sg_commit(); the view stays valid until the next render().
        sg_view render(int width, int height);

    private:
        // (Re)creates the pixel buffer and output target for a new size and maps the buffer
        // into CUDA. False on failure (logged); the renderer then returns an empty view.
        bool rebuildTargets(int width, int height);
        // Blocks until neither CUDA nor the graphics queue is using the pixel buffer.
        void waitIdle();

        RaymarchParams m_params{}; // everything the kernel gets, kept current by the setters above

        // Released explicitly in the destructor, before the Vulkan buffer and semaphores it
        // imported are destroyed.
        std::optional<gpu::CudaInterop> m_cuda;
        CUmodule m_module = nullptr;
        CUfunction m_raymarch = nullptr;

        std::optional<gpu::CrossSemaphore> m_cuda_done;   // CUDA -> graphics queue: pixels written
        std::optional<gpu::CrossSemaphore> m_render_done; // graphics queue -> CUDA: pixels read
        int m_cuda_done_sem = -1;
        int m_render_done_sem = -1;
        u64 m_frame = 0; // frames rendered; the semaphore value for this frame

        int m_width = 0;
        int m_height = 0;
        std::optional<gpu::ExternalBuffer> m_pixels; // width * height RGBA8, written by CUDA
        CUdeviceptr m_pixels_cuda = 0;
        gpu::ViewHandle m_pixels_view;               // storage-buffer view for the copy pass
        graphics::primitives::pass::RenderTarget m_target;

        graphics::primitives::pass::FullscreenPass m_copy;
        bool m_ready = false; // CUDA, the kernel and the semaphores all came up
};

static_assert(graphics::Renderer<ElectronCloudRenderer>);

} // namespace renderer
