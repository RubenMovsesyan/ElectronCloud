#include <renderer/electron_cloud_renderer.hpp>

#include <common.h>
#include <rlog.h>

#include <array>
#include <cmath>

namespace renderer {

namespace {

// Written by build.cpp's nvcc step; relative to the project root, where ./run starts us.
constexpr const char* kPtxPath = ".build/cuda/electron_cloud_renderer.ptx";

constexpr sg_color kPlaceholderColor = {1.0f, 1.0f, 0.0f, 1.0f};

} // namespace

ElectronCloudRenderer::ElectronCloudRenderer() : m_placeholder(kPlaceholderColor) {
    // Makes a context current on this (the render) thread, on sokol's GPU.
    m_cuda = gpu::CudaInterop::create();
    if (!m_cuda) {
        RLOG(LL_ERROR, "ElectronCloudRenderer: CUDA interop failed to start");
        return;
    }

    CUdevice device = 0;
    std::array<char, 256> name{};
    cuCtxGetDevice(&device);
    cuDeviceGetName(name.data(), static_cast<int>(name.size()), device);
    RLOG(LL_INFO, "ElectronCloudRenderer: CUDA on %s", name.data());

    if (cuModuleLoad(&m_module, kPtxPath) != CUDA_SUCCESS || cuModuleGetFunction(&m_fill, m_module, "fill") != CUDA_SUCCESS) {
        RLOG(LL_ERROR, "ElectronCloudRenderer: failed to load %s", kPtxPath);
        return;
    }

    if (smokeTest()) {
        RLOG(LL_INFO, "ElectronCloudRenderer: CUDA kernel smoke test passed");
    }
}

ElectronCloudRenderer::~ElectronCloudRenderer() {
    if (m_module != nullptr) {
        cuModuleUnload(m_module);
    }
}

bool ElectronCloudRenderer::smokeTest() {
    constexpr u32 kCount = 256;
    constexpr f32 kScale = 0.5f;

    CUdeviceptr device_out = 0;
    if (cuMemAlloc(&device_out, kCount * sizeof(f32)) != CUDA_SUCCESS) {
        RLOG(LL_ERROR, "ElectronCloudRenderer: smoke test cuMemAlloc failed");
        return false;
    }

    u32 count = kCount;
    f32 scale = kScale;
    void* args[] = {&device_out, &count, &scale};
    const u32 block = 128;
    const u32 grid = (kCount + block - 1) / block;

    std::array<f32, kCount> host{};
    const bool ran = cuLaunchKernel(m_fill, grid, 1, 1, block, 1, 1, 0, m_cuda->stream(), args, nullptr) == CUDA_SUCCESS &&
                     cuStreamSynchronize(m_cuda->stream()) == CUDA_SUCCESS &&
                     cuMemcpyDtoH(host.data(), device_out, sizeof(host)) == CUDA_SUCCESS;
    cuMemFree(device_out);

    if (!ran) {
        RLOG(LL_ERROR, "ElectronCloudRenderer: smoke test launch failed");
        return false;
    }
    for (u32 i = 0; i < kCount; i++) {
        if (std::fabs(host[i] - kScale * static_cast<f32>(i)) > 1e-6f) {
            RLOG(LL_ERROR, "ElectronCloudRenderer: smoke test wrong at %u: %f", i, host[i]);
            return false;
        }
    }
    return true;
}

sg_view ElectronCloudRenderer::render(int width, int height) {
    return m_placeholder.render(width, height);
}

} // namespace renderer
