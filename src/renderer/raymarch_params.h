#pragma once

// Plain data shared by the C++ renderer and the CUDA kernels (electron_cloud_renderer.cu),
// passed by value to cuLaunchKernel. Only floats and unsigneds, no Eigen and no CUDA vector
// types, so both compilers lay it out identically.

struct RayCamera {
    float position[3]; // where every ray starts (world space)
    float forward[3];  // unit view direction
    float right[3];    // unit, screen +x
    float up[3];       // unit, screen +y
    float tan_half_fov_y;
    float aspect;      // width / height
};

struct RaySphere {
    float center[3];
    float radius;
};

struct RaymarchParams {
    RayCamera camera;
    RaySphere sphere;
    float step;         // distance advanced per march step (delta t)
    float max_distance; // a ray gives up this far from the camera
    unsigned width;     // output size in pixels; the pixel buffer is width * height RGBA8
    unsigned height;
};
