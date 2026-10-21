#pragma once

#ifndef GAMEENGINE_CUDA
#error "ElectronCloudRenderer needs CUDA: build.cpp defines GAMEENGINE_CUDA when it finds the toolkit"
#endif

#include <gpu/cuda_interop.hpp>
#include <graphics/renderer.hpp>
#include <graphics/renderers/clear_renderer.hpp>

#include <cuda.h>

#include <optional>

namespace renderer {

// ElectronCloud's own implementation of GameEngine's graphics::Renderer concept. Lives here,
// not in the engine: it plugs into graphics::Presenter like any engine renderer, but it is
// specific to this project.
//
// CUDA runs inside the renderer, on the render thread: the constructor brings up a CUDA
// context on the same GPU sokol's Vulkan device uses and loads this renderer's kernels
// (electron_cloud_renderer.cu, compiled to PTX by build.cpp). The pixel path -- CUDA
// computing the image render() returns -- is not wired yet; render() clears to a solid
// colour until it is.
//
// sokol_gfx is single-threaded: construct, render and destroy on the Application's thread.
class ElectronCloudRenderer {
    public:
        ElectronCloudRenderer();
        ~ElectronCloudRenderer();

        ElectronCloudRenderer(const ElectronCloudRenderer&) = delete;
        ElectronCloudRenderer& operator=(const ElectronCloudRenderer&) = delete;

        // Records this frame's passes at width x height pixels and returns a view of the
        // result. Does not call sg_commit(); the view stays valid until the next render().
        sg_view render(int width, int height);

    private:
        // Runs a tiny kernel and checks the result on the CPU: proves nvcc, the PTX load and
        // a launch all work. Logs and returns false on any failure.
        bool smokeTest();

        // Declared first so it is destroyed last: the module belongs to its context.
        std::optional<gpu::CudaInterop> m_cuda;
        CUmodule m_module = nullptr;
        CUfunction m_fill = nullptr;

        graphics::renderers::ClearRenderer m_placeholder;
};

static_assert(graphics::Renderer<ElectronCloudRenderer>);

} // namespace renderer
