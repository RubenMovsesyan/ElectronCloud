// ElectronCloudRenderer's CUDA kernels. build.cpp compiles this file to PTX
// (.build/cuda/electron_cloud_renderer.ptx, nvcc --ptx); the renderer JIT-loads it at
// runtime with cuModuleLoad, so nvcc never touches the C++ link.

// Smoke test: out[i] = scale * i. Checked on the CPU once at startup.
extern "C" __global__ void fill(float* out, unsigned count, float scale) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = scale * static_cast<float>(i);
    }
}
