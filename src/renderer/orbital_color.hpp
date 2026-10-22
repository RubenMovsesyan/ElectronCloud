#pragma once

// Each orbital's colour: the orbitals that have a wave function are spaced evenly around the
// hue wheel, in Orbital order (1s at red). Purely cosmetic; host-side.

#include <renderer/hydrogen.hpp>
#include <renderer/orbital.hpp>

#include <common.h>

#include <Eigen/Core>

#include <cmath>

namespace renderer {

constexpr f32 kOrbitalSaturation = 0.7f;
constexpr f32 kOrbitalValue = 1.0f;

// HSV -> linear RGB, hue in turns (0 .. 1 is once round the wheel).
inline Eigen::Vector3f hsvToRgb(f32 hue, f32 saturation, f32 value) {
    const f32 h = (hue - std::floor(hue)) * 6.0f;
    const f32 chroma = value * saturation;
    const f32 x = chroma * (1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f));
    Eigen::Vector3f rgb;
    switch (static_cast<i32>(h) % 6) {
        case 0: rgb = {chroma, x, 0.0f}; break;
        case 1: rgb = {x, chroma, 0.0f}; break;
        case 2: rgb = {0.0f, chroma, x}; break;
        case 3: rgb = {0.0f, x, chroma}; break;
        case 4: rgb = {x, 0.0f, chroma}; break;
        default: rgb = {chroma, 0.0f, x}; break;
    }
    return rgb + Eigen::Vector3f::Constant(value - chroma);
}

// Where `orbital` sits on the hue wheel, in turns: its position among the orbitals that have a
// wave function, divided by how many there are. 0 for one without.
constexpr f32 orbitalHue(Orbital orbital) {
    u32 index = 0;
    u32 count = 0;
    for (u32 i = 0; i < static_cast<u32>(Orbital::Count); i++) {
        if (hasWaveFunction(static_cast<Orbital>(i))) {
            if (static_cast<Orbital>(i) == orbital) {
                index = count;
            }
            count++;
        }
    }
    return count > 0 ? static_cast<f32>(index) / static_cast<f32>(count) : 0.0f;
}

inline Eigen::Vector3f orbitalColor(Orbital orbital) {
    return hsvToRgb(orbitalHue(orbital), kOrbitalSaturation, kOrbitalValue);
}

// The colour a fraction `t` (0 .. 1) of the way from `from`'s colour to `to`'s, turning the hue
// the shorter way round the wheel so it stays saturated (blending RGB would pass through grey).
inline Eigen::Vector3f blendOrbitalColors(Orbital from, Orbital to, f32 t) {
    const f32 start = orbitalHue(from);
    f32 delta = orbitalHue(to) - start;
    delta -= std::round(delta); // -0.5 .. 0.5: the shorter way
    return hsvToRgb(start + delta * t, kOrbitalSaturation, kOrbitalValue);
}

} // namespace renderer
