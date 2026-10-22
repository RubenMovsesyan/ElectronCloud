#pragma once

// Hydrogen-like wave functions psi_nlm(r, theta, phi), the real orbitals named by
// renderer::Orbital. Compiled into both the CUDA kernel and host code (so they can be checked
// on the CPU), hence HYDROGEN_HD.
//
// Everything is in terms of rho = r / a, where a = a0 / Z is the Bohr radius scaled by the
// nuclear charge, in world units. The prefactor 1 / sqrt(pi a^3) carries the units, so
// |psi|^2 is a probability density per world unit cubed and integrates to 1 over all space.
//
// These return psi itself, not |psi|^2: a superposition adds the wave functions of its states
// before squaring. Real orbitals are real-valued, so |psi|^2 = psi * psi.

#include <renderer/orbital.hpp>

#include <common.h>

#include <Eigen/Core>

#include <cmath>

#if defined(__CUDA__) || defined(__CUDACC__)
#define HYDROGEN_HD __host__ __device__
#else
#define HYDROGEN_HD
#endif

namespace renderer {

constexpr f32 kPi = 3.14159265358979323846f;

// A point relative to the nucleus in spherical coordinates (physics convention: `theta` the
// polar angle down from +z, `phi` the azimuth from +x towards +y). The angles are kept as
// their sines and cosines -- all the wave functions use -- which come straight from the
// Cartesian offset (cos theta = z / r, sin theta cos phi = x / r, ...), so no acos / atan2 /
// sin / cos runs per sample. That was ~3.7x the cost of the whole 1s march.
struct Spherical {
    f32 r;
    f32 cos_theta;
    f32 sin_theta;
    f32 cos_phi;
    f32 sin_phi;
};

HYDROGEN_HD inline Spherical toSpherical(const Eigen::Vector3f& offset) {
    const f32 r = offset.norm();
    const f32 rho_xy = sqrtf(offset.x() * offset.x() + offset.y() * offset.y()); // r sin theta
    // At the nucleus (and on the z axis, for phi) the angles are undefined; any value works,
    // since every orbital with an angular part also has a radial factor that is 0 there.
    const f32 cos_theta = r > 0.0f ? offset.z() / r : 1.0f;
    const f32 sin_theta = r > 0.0f ? rho_xy / r : 0.0f;
    const f32 cos_phi = rho_xy > 0.0f ? offset.x() / rho_xy : 1.0f;
    const f32 sin_phi = rho_xy > 0.0f ? offset.y() / rho_xy : 0.0f;
    return Spherical{r, cos_theta, sin_theta, cos_phi, sin_phi};
}

// Whether waveFunction() implements `orbital` (everything up to n = 3 so far).
constexpr bool hasWaveFunction(Orbital orbital) {
    return static_cast<u32>(orbital) <= static_cast<u32>(Orbital::D3x2_y2);
}

// psi for `orbital` at `point`, with a = a0 / Z in world units. 0 for orbitals not implemented
// yet (n = 4) and for Orbital::Count. m -> real orbital follows the convention in orbital.hpp:
// m > 0 is the cos(|m| phi) one, m < 0 the sin(|m| phi) one.
HYDROGEN_HD inline f32 waveFunction(Orbital orbital, const Spherical& point, f32 a) {
    const f32 rho = point.r / a;
    const f32 norm = 1.0f / sqrtf(kPi * a * a * a); // 1 / sqrt(pi a^3)

    const f32 cos_theta = point.cos_theta;
    const f32 sin_theta = point.sin_theta;
    const f32 cos_phi = point.cos_phi;
    const f32 sin_phi = point.sin_phi;

    switch (orbital) {
        // n = 1
        case Orbital::S1:
            return norm * expf(-rho);

        // n = 2: common factor 1 / (4 sqrt(2 pi a^3)), e^(-rho / 2)
        case Orbital::S2:
        case Orbital::P2y:
        case Orbital::P2z:
        case Orbital::P2x: {
            const f32 radial = norm / (4.0f * sqrtf(2.0f)) * expf(-rho / 2.0f);
            switch (orbital) {
                case Orbital::S2: return radial * (2.0f - rho);
                case Orbital::P2z: return radial * rho * cos_theta;
                case Orbital::P2x: return radial * rho * sin_theta * cos_phi;
                default: return radial * rho * sin_theta * sin_phi; // P2y
            }
        }

        // n = 3: e^(-rho / 3)
        case Orbital::S3:
            return norm / (81.0f * sqrtf(3.0f)) * (27.0f - 18.0f * rho + 2.0f * rho * rho) * expf(-rho / 3.0f);

        case Orbital::P3y:
        case Orbital::P3z:
        case Orbital::P3x: {
            const f32 radial = norm * sqrtf(2.0f) / 81.0f * rho * (6.0f - rho) * expf(-rho / 3.0f);
            switch (orbital) {
                case Orbital::P3z: return radial * cos_theta;
                case Orbital::P3x: return radial * sin_theta * cos_phi;
                default: return radial * sin_theta * sin_phi; // P3y
            }
        }

        case Orbital::D3z2:
            return norm / (81.0f * sqrtf(6.0f)) * rho * rho * expf(-rho / 3.0f) * (3.0f * cos_theta * cos_theta - 1.0f);

        case Orbital::D3xz:
        case Orbital::D3yz: {
            const f32 radial = norm * sqrtf(2.0f) / 81.0f * rho * rho * expf(-rho / 3.0f) * sin_theta * cos_theta;
            return orbital == Orbital::D3xz ? radial * cos_phi : radial * sin_phi;
        }

        // 1 / (81 sqrt(2 pi a^3)): half the d_xz / d_yz prefactor, since the integral of
        // (sin^2 theta cos 2phi)^2 over the sphere is 4x that of (sin theta cos theta cos phi)^2.
        // (Often misprinted as 1 / (162 sqrt(pi a^3)), which integrates to 1/2.)
        case Orbital::D3x2_y2:
        case Orbital::D3xy: {
            const f32 radial = norm / (81.0f * sqrtf(2.0f)) * rho * rho * expf(-rho / 3.0f) * sin_theta * sin_theta;
            // cos 2phi = cos^2 phi - sin^2 phi, sin 2phi = 2 sin phi cos phi
            return orbital == Orbital::D3x2_y2 ? radial * (cos_phi * cos_phi - sin_phi * sin_phi) : radial * (2.0f * sin_phi * cos_phi);
        }

        default:
            return 0.0f;
    }
}

// The largest |psi|^2 anywhere for `orbital`, with a = a0 / Z in world units -- what the
// renderer divides by so every orbital uses the full brightness range. Host-only (it's run
// once per orbital change, not per sample); 0 for orbitals without a wave function.
//
// Every orbital here is radial(r) * angular(theta, phi), so the peak is the largest angular
// value combined with the largest radial value along that direction: find the brightest
// direction on a sphere of radius r0 (any r0 where the radial part isn't 0), then the
// brightest radius along it. ~20k evaluations, well under a millisecond.
inline f32 peakDensity(Orbital orbital, f32 a) {
    if (!hasWaveFunction(orbital)) {
        return 0.0f;
    }
    auto density = [&](const Eigen::Vector3f& offset) {
        const f32 psi = waveFunction(orbital, toSpherical(offset), a);
        return psi * psi;
    };

    // 1. Brightest direction. 1.2345 a is clear of every radial node up to n = 3
    //    (2s: 2a; 3s: 1.9a, 7.1a; 3p: 6a).
    constexpr i32 kThetaSteps = 90;
    constexpr i32 kPhiSteps = 180;
    const f32 r0 = 1.2345f * a;
    Eigen::Vector3f best_direction = Eigen::Vector3f::UnitZ();
    f32 best = -1.0f;
    for (i32 i = 0; i <= kThetaSteps; i++) {
        const f32 theta = kPi * static_cast<f32>(i) / kThetaSteps;
        for (i32 j = 0; j < kPhiSteps; j++) {
            const f32 phi = 2.0f * kPi * static_cast<f32>(j) / kPhiSteps;
            const Eigen::Vector3f direction{sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)};
            const f32 value = density(direction * r0);
            if (value > best) {
                best = value;
                best_direction = direction;
            }
        }
    }

    // 2. Brightest radius along it, from the nucleus out past where any n <= 3 orbital
    //    matters (~30 a).
    constexpr i32 kRadialSteps = 6000;
    f32 peak = 0.0f;
    for (i32 i = 0; i <= kRadialSteps; i++) {
        const f32 r = 30.0f * a * static_cast<f32>(i) / kRadialSteps;
        peak = fmaxf(peak, density(best_direction * r));
    }
    return peak;
}

} // namespace renderer
