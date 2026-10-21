#pragma once

// Plain data shared by the C++ renderer and the CUDA kernels (electron_cloud_renderer.cu),
// passed by value to cuLaunchKernel. Only f32s and unsigneds, no Eigen and no CUDA vector
// types, so both compilers lay it out identically.

#include <common.h>

struct RayCamera {
    f32 position[3]; // where every ray starts (world space)
    f32 forward[3];  // unit view direction
    f32 right[3];    // unit, screen +x
    f32 up[3];       // unit, screen +y
    f32 tan_half_fov_y;
    f32 aspect;      // width / height
};

struct RaySphere {
    f32 center[3];
    f32 radius;
};

struct RaymarchParams {
    RayCamera camera;
    RaySphere sphere;
    f32 step;         // distance advanced per march step (delta t)
    f32 max_distance; // a ray gives up this far from the camera
    unsigned width;     // output size in pixels; the pixel buffer is width * height RGBA8
    unsigned height;
};
