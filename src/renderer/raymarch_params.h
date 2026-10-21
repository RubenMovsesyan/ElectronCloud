#pragma once

// Plain data shared by the C++ renderer and the CUDA kernels (electron_cloud_renderer.cu),
// passed by value to cuLaunchKernel. Only fixed-size scalars from common.h and arrays of
// them -- no Eigen, no CUDA vector types -- so both compilers lay it out identically.
//
// Units: positions and lengths are world units unless noted; the wave function itself works
// in atomic units (lengths in Bohr radii a0, time in hbar / E_h ~= 2.419e-17 s), and
// RayAtom::bohr_radius converts between the two.

#include <common.h>

#include <stddef.h>

struct RayCamera {
    f32 position[3]; // where every ray starts
    f32 forward[3];  // unit view direction
    f32 right[3];    // unit, screen +x
    f32 up[3];       // unit, screen +y
    f32 tan_half_fov_y;
    f32 aspect;      // width / height
};

// One hydrogen-like eigenstate |n l m>, weighted by a complex coefficient
// amplitude * e^(i * phase) in the superposition.
struct RayQuantumState {
    i32 n; // principal, >= 1
    i32 l; // azimuthal, 0 .. n-1
    i32 m; // magnetic, -l .. l
    f32 amplitude;
    f32 phase; // radians, at time 0
};

// Everything the density depends on: a nucleus and the electron's state around it.
// A transition between energy levels is a superposition of the start and end states whose
// amplitudes the app moves over time; a stationary state uses only states[0].
struct RayAtom {
    f32 nucleus[3];     // world-space position of the nucleus
    f32 nuclear_charge; // Z: 1 = hydrogen; hydrogen-like ions (He+, Li2+, ...) above that
    f32 bohr_radius;    // world units per Bohr radius -- sets the cloud's size on screen
    f32 bounds_radius;  // world units around the nucleus; density is taken as 0 outside
    RayQuantumState states[2];
    u32 state_count; // 1 or 2
};

// How rays are marched. Trades quality for speed; doesn't change what the cloud is.
struct RayMarch {
    f32 step_size;      // distance advanced per sample
    f32 max_distance;   // a ray gives up this far from the camera
    f32 min_density;    // samples at or below this are treated as empty space
    f32 opacity_cutoff; // a ray stops once this opaque; nothing behind it would show
};

// Purely cosmetic: how density turns into colour.
struct RayAppearance {
    f32 cloud_color[3];      // colour of density 1; scaled by density per sample
    f32 background_color[3]; // shows through wherever the cloud isn't opaque
    f32 absorption;          // Beer-Lambert strength: a sample is 1 - exp(-density * step_size * absorption) opaque
};

struct RaymarchParams {
    RayCamera camera;
    RayAtom atom;
    RayMarch march;
    RayAppearance appearance;
    f64 time;   // simulation time, atomic units of time (hbar / E_h); f64 so phases stay exact as it grows
    u32 width;  // output size in pixels; the pixel buffer is width * height RGBA8
    u32 height;
};

// Both nvcc and clang compile this header; if either lays a struct out differently, the
// kernel would read garbage. Update these when a field changes.
static_assert(sizeof(RayCamera) == 56, "RayCamera layout changed");
static_assert(sizeof(RayAtom) == 68, "RayAtom layout changed");
static_assert(sizeof(RayMarch) == 16, "RayMarch layout changed");
static_assert(sizeof(RayAppearance) == 28, "RayAppearance layout changed");
static_assert(offsetof(RaymarchParams, time) == 168, "RaymarchParams layout changed");
static_assert(sizeof(RaymarchParams) == 184, "RaymarchParams layout changed");
