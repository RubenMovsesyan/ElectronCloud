#pragma once

// What the renderer draws and how -- one set of types, used by the app (src/main.cpp), the
// C++ renderer and the CUDA kernels alike. RaymarchParams is passed by value to
// cuLaunchKernel, so both compilers must lay these out identically: stick to fixed-size
// scalars from common.h, Eigen::Vector3f (three packed f32, no alignment) and std::array of
// those. No Eigen type with 16-byte alignment (Vector4f, Matrix4f): it would move fields.
// The static_asserts at the end catch a mismatch.
//
// Units: positions and lengths are world units unless noted; the wave function itself works
// in atomic units (lengths in Bohr radii a0, time in hbar / E_h ~= 2.419e-17 s), and
// Atom::bohr_radius converts between the two.

#include <common.h>

#include <Eigen/Core>

#include <array>
#include <cstddef>

namespace renderer {

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
    f32 nuclear_charge = 1.0f;                          // Z: 1 = hydrogen; hydrogen-like ions above that
    f32 bohr_radius = 0.529f;                           // world units per Bohr radius; 0.529 = world units are angstroms
    f32 bounds_radius = 2.5f;                           // world units around the nucleus; density is 0 outside
    std::array<QuantumState, 2> states{};
    u32 state_count = 1; // 1 or 2
};

// How rays are marched: quality against speed. Doesn't change what the cloud is.
struct RaymarchSettings {
    f32 step_size = 0.1f;       // distance advanced per sample
    f32 max_distance = 10.0f;   // a ray gives up this far from the camera
    f32 min_density = 0.001f;   // samples at or below this are treated as empty space
    f32 opacity_cutoff = 0.95f; // a ray stops once this opaque; nothing behind it would show
};

// Purely cosmetic: how density turns into colour.
struct Appearance {
    Eigen::Vector3f cloud_color{0.3f, 0.6f, 1.0f};        // colour of density 1; scaled by density per sample
    Eigen::Vector3f background_color{0.02f, 0.02f, 0.03f}; // shows through wherever the cloud isn't opaque
    f32 absorption = 10.0f; // Beer-Lambert strength: a sample is 1 - exp(-density * step_size * absorption) opaque
};

// Where the rays start and how they fan out. Built by the renderer from the engine's Camera
// each frame (beginFrame), not set by the app.
struct RayCamera {
    Eigen::Vector3f position = Eigen::Vector3f::Zero(); // where every ray starts
    Eigen::Vector3f forward = -Eigen::Vector3f::UnitZ(); // unit view direction
    Eigen::Vector3f right = Eigen::Vector3f::UnitX();    // unit, screen +x
    Eigen::Vector3f up = Eigen::Vector3f::UnitY();       // unit, screen +y
    f32 tan_half_fov_y = 0.57735f;
    f32 aspect = 1.0f; // width / height
};

// Everything one launch of the raymarch kernel gets.
struct RaymarchParams {
    RayCamera camera;
    Atom atom;
    RaymarchSettings march;
    Appearance appearance;
    f64 time = 0.0; // simulation time, atomic units of time (hbar / E_h); f64 so phases stay exact as it grows
    u32 width = 0;  // output size in pixels; the pixel buffer is width * height RGBA8
    u32 height = 0;
};

// The host compiler and the CUDA device compiler both check these; if either lays a struct
// out differently, the build fails instead of the kernel reading garbage. Update them when a
// field changes.
static_assert(sizeof(Eigen::Vector3f) == 12, "Eigen::Vector3f must be three packed f32");
static_assert(sizeof(RayCamera) == 56, "RayCamera layout changed");
static_assert(sizeof(Atom) == 68, "Atom layout changed");
static_assert(sizeof(RaymarchSettings) == 16, "RaymarchSettings layout changed");
static_assert(sizeof(Appearance) == 28, "Appearance layout changed");
static_assert(offsetof(RaymarchParams, time) == 168, "RaymarchParams layout changed");
static_assert(sizeof(RaymarchParams) == 184, "RaymarchParams layout changed");

} // namespace renderer
