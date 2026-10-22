#pragma once

// The physics of driving the electron from one orbital to another with light (Rabi
// oscillation), done on the CPU in f64. The kernel only ever sees the resulting
// renderer::Superposition weights.
//
// Atomic units throughout (hbar = m_e = e = a0 = 1): energies in Hartree, time in
// hbar / E_h ~= 2.419e-17 s, dipoles in e a0, fields in E_h / (e a0) ~= 5.142e11 V/m.
//
// Driven on resonance by light polarised along the transition dipole d, with field
// amplitude E0, the state is Psi = c_A psi_A e^(-i E_A t) + c_B psi_B e^(-i E_B t) with
//   c_A = cos(Omega t / 2),  c_B = -i sin(Omega t / 2),  Omega = |d| E0 (the Rabi frequency),
// so the density is
//   |Psi|^2 = cos^2(Omega t/2) psi_A^2 + sin^2(Omega t/2) psi_B^2
//             - sin(Omega t) sin(omega0 t) psi_A psi_B,      omega0 = E_B - E_A.
// The transition is complete (all population in B) at t = pi / Omega. The last term makes the
// cloud slosh at omega0; it happens omega0 / (2 Omega) times per transition.

#include <renderer/hydrogen.hpp>
#include <renderer/orbital.hpp>
#include <renderer/raymarch_params.h>

#include <common.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <numbers>

namespace renderer {

// 1 atomic unit of intensity: the intensity of a wave whose peak field is 1 E_h / (e a0).
constexpr f64 kAtomicUnitIntensityWattsPerCm2 = 3.50944758e16;
// Atomic unit of time in seconds (hbar / E_h).
constexpr f64 kAtomicUnitTimeSeconds = 2.4188843265857e-17;
// The dipole a forbidden transition is animated with, as if it were allowed (e a0). In real
// life its dipole is 0 and the light would never move it.
constexpr f64 kForbiddenStandInDipole = 1.0;

// Hydrogen-like energy of shell n, in Hartree: -Z^2 / (2 n^2).
inline f64 energyHartree(i32 n, f64 nuclear_charge) {
    return -nuclear_charge * nuclear_charge / (2.0 * n * n);
}

// <psi_A | r | psi_B> for every pair of orbitals with a wave function, for hydrogen (Z = 1),
// in e a0. For nuclear charge Z, divide by Z (every length scales with a0 / Z).
//
// Computed numerically once, on first use (~0.2 s), from hydrogen.hpp's own wave functions,
// so the selection rules it implies are exactly those of the orbitals the renderer draws.
// Only pairs with l differing by 1 are integrated: for every other pair the dipole is 0 by
// the angular-momentum selection rule (an exact result, not an approximation).
class DipoleTable {
    public:
        static const DipoleTable& get() {
            static const DipoleTable table;
            return table;
        }

        const Eigen::Vector3d& dipole(Orbital a, Orbital b) const {
            return m_dipoles[static_cast<u32>(a)][static_cast<u32>(b)];
        }

        // Whether light can drive a <-> b at all: its dipole is non-zero (relative to the
        // quadrature's noise; the smallest allowed dipole up to n = 3 is ~0.2 e a0).
        bool allowed(Orbital a, Orbital b) const { return dipole(a, b).norm() > 1e-3; }

    private:
        static constexpr u32 kCount = static_cast<u32>(Orbital::Count);

        DipoleTable() {
            for (auto& row : m_dipoles) {
                row.fill(Eigen::Vector3d::Zero());
            }

            // Pairs worth integrating: both have wave functions, l differs by exactly 1.
            struct Pair {
                    u32 a;
                    u32 b;
            };
            std::array<Pair, kCount * kCount> pairs{};
            u32 pair_count = 0;
            std::array<u32, kCount> used{};
            u32 used_count = 0;
            for (u32 a = 0; a < kCount; a++) {
                if (hasWaveFunction(static_cast<Orbital>(a))) {
                    used[used_count++] = a;
                }
                for (u32 b = a + 1; b < kCount; b++) {
                    const Orbital oa = static_cast<Orbital>(a);
                    const Orbital ob = static_cast<Orbital>(b);
                    if (hasWaveFunction(oa) && hasWaveFunction(ob) && std::abs(quantumNumbersOf(oa).l - quantumNumbersOf(ob).l) == 1) {
                        pairs[pair_count++] = Pair{a, b};
                    }
                }
            }

            // Midpoint quadrature in spherical coordinates, a = 1 (Bohr radii), out to 40 a0
            // (n = 3 densities are ~1e-8 of their peak there).
            constexpr i32 kRadial = 320;
            constexpr i32 kPolar = 48;
            constexpr i32 kAzimuthal = 96;
            constexpr f64 kRadius = 40.0;
            constexpr f64 dr = kRadius / kRadial;
            constexpr f64 dtheta = std::numbers::pi / kPolar;
            constexpr f64 dphi = 2.0 * std::numbers::pi / kAzimuthal;

            std::array<f64, kCount> psi{};
            std::array<Eigen::Vector3d, kCount * kCount> sums{};
            sums.fill(Eigen::Vector3d::Zero());
            for (i32 i = 0; i < kRadial; i++) {
                const f64 r = (i + 0.5) * dr;
                for (i32 j = 0; j < kPolar; j++) {
                    const f64 theta = (j + 0.5) * dtheta;
                    const f64 weight = r * r * std::sin(theta) * dr * dtheta * dphi;
                    for (i32 k = 0; k < kAzimuthal; k++) {
                        const f64 phi = (k + 0.5) * dphi;
                        const Eigen::Vector3d position{r * std::sin(theta) * std::cos(phi), r * std::sin(theta) * std::sin(phi), r * std::cos(theta)};
                        const Spherical point = toSpherical(position.cast<f32>());
                        for (u32 u = 0; u < used_count; u++) {
                            psi[used[u]] = waveFunction(static_cast<Orbital>(used[u]), point, 1.0f);
                        }
                        const Eigen::Vector3d weighted = position * weight;
                        for (u32 p = 0; p < pair_count; p++) {
                            sums[p] += weighted * (psi[pairs[p].a] * psi[pairs[p].b]);
                        }
                    }
                }
            }
            for (u32 p = 0; p < pair_count; p++) {
                m_dipoles[pairs[p].a][pairs[p].b] = sums[p];
                m_dipoles[pairs[p].b][pairs[p].a] = sums[p];
            }
        }

        std::array<std::array<Eigen::Vector3d, kCount>, kCount> m_dipoles;
};

// Everything fixed about one A -> B transition, worked out when it starts.
struct TransitionPlan {
    f64 omega0 = 0.0;        // E_B - E_A (Hartree = rad per atomic time); negative when B is lower (emission)
    f64 rabi = 0.0;          // Omega = |d| E0
    f64 duration = 0.0;      // pi / Omega, atomic units of time: when all population is in B
    f64 dipole = 0.0;        // |d|, e a0 -- the stand-in dipole for a forbidden transition
    f64 sloshes = 0.0;       // |omega0| / (2 Omega): how many times the cloud sloshes on the way
    bool allowed = false;    // false: forbidden for light, animated with kForbiddenStandInDipole
};

// Plans `from` -> `to` driven by light of `intensity_watts_per_cm2`, polarised along the
// transition dipole, for an atom of `nuclear_charge`.
inline TransitionPlan planTransition(Orbital from, Orbital to, f64 nuclear_charge, f64 intensity_watts_per_cm2) {
    const DipoleTable& table = DipoleTable::get();
    TransitionPlan plan;
    plan.allowed = table.allowed(from, to);
    plan.dipole = plan.allowed ? table.dipole(from, to).norm() / nuclear_charge : kForbiddenStandInDipole;
    plan.omega0 = energyHartree(quantumNumbersOf(to).n, nuclear_charge) - energyHartree(quantumNumbersOf(from).n, nuclear_charge);

    const f64 field = std::sqrt(intensity_watts_per_cm2 / kAtomicUnitIntensityWattsPerCm2); // E0, atomic units
    plan.rabi = plan.dipole * field;
    plan.duration = plan.rabi > 0.0 ? std::numbers::pi / plan.rabi : 0.0;
    plan.sloshes = plan.rabi > 0.0 ? std::abs(plan.omega0) / (2.0 * plan.rabi) : 0.0;
    return plan;
}

// The density weights at time `t` (atomic units, from the start of the transition).
inline Superposition superpositionAt(const TransitionPlan& plan, f64 t) {
    const f64 half = 0.5 * plan.rabi * t;
    return Superposition{
        .a = static_cast<f32>(std::cos(half) * std::cos(half)),
        .b = static_cast<f32>(std::sin(half) * std::sin(half)),
        .interference = static_cast<f32>(-std::sin(plan.rabi * t) * std::sin(plan.omega0 * t)),
    };
}

} // namespace renderer
