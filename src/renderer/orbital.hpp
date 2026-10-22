#pragma once

#include <common.h>

#include <optional>

namespace renderer {

// The individual spatial orbitals up to n = 4, named by their real-orbital designation.
// Enumerator names spell the designation shell-letter first, since an identifier can't start
// with a digit: D3x2_y2 is 3d_{x^2 - y^2}, F4y_3x2_y2 is 4f_{y(3x^2 - y^2)}.
//
// Declared in table order -- n, then l, then m ascending -- which orbitalFor() relies on.
//
// Convention: the usual real-orbital one. For m != 0 the real orbital is a combination of the
// complex +m and -m states: m < 0 is the sin(|m| phi) one, m > 0 the cos(|m| phi) one. So
// m -1 is p_y / d_yz / f_yz^2, m +1 is p_x / d_xz / f_xz^2, m -2 is d_xy / f_xyz, m +2 is
// d_x^2-y^2 / f_z(x^2-y^2), m -3 is f_y(3x^2-y^2), m +3 is f_x(x^2-3y^2). The wave function
// must use the same convention when it picks an angular formula from an Orbital.
enum class Orbital : u32 {
    // n = 1
    S1, // 1s       (l 0, m 0)

    // n = 2
    S2,  // 2s      (l 0, m 0)
    P2y, // 2p_y    (l 1, m -1)
    P2z, // 2p_z    (l 1, m  0)
    P2x, // 2p_x    (l 1, m +1)

    // n = 3
    S3,      // 3s           (l 0, m 0)
    P3y,     // 3p_y         (l 1, m -1)
    P3z,     // 3p_z         (l 1, m  0)
    P3x,     // 3p_x         (l 1, m +1)
    D3xy,    // 3d_xy        (l 2, m -2)
    D3yz,    // 3d_yz        (l 2, m -1)
    D3z2,    // 3d_z^2       (l 2, m  0)
    D3xz,    // 3d_xz        (l 2, m +1)
    D3x2_y2, // 3d_x^2-y^2   (l 2, m +2)

    // n = 4
    S4,         // 4s               (l 0, m 0)
    P4y,        // 4p_y             (l 1, m -1)
    P4z,        // 4p_z             (l 1, m  0)
    P4x,        // 4p_x             (l 1, m +1)
    D4xy,       // 4d_xy            (l 2, m -2)
    D4yz,       // 4d_yz            (l 2, m -1)
    D4z2,       // 4d_z^2           (l 2, m  0)
    D4xz,       // 4d_xz            (l 2, m +1)
    D4x2_y2,    // 4d_x^2-y^2       (l 2, m +2)
    F4y_3x2_y2, // 4f_y(3x^2-y^2)   (l 3, m -3)
    F4xyz,      // 4f_xyz           (l 3, m -2)
    F4yz2,      // 4f_yz^2          (l 3, m -1)
    F4z3,       // 4f_z^3           (l 3, m  0)
    F4xz2,      // 4f_xz^2          (l 3, m +1)
    F4z_x2_y2,  // 4f_z(x^2-y^2)    (l 3, m +2)
    F4x_x2_3y2, // 4f_x(x^2-3y^2)   (l 3, m +3)

    Count,
};

// The orbital |n l m> names, or nullopt when the quantum numbers are invalid (n < 1, l outside
// 0 .. n-1, m outside -l .. l) or past the table (n > 4).
//
// Shell n starts after every orbital of the shells below it (1 + 4 + 9 + ... = sum of k^2 for
// k < n), subshell l after the l^2 orbitals of the subshells below it in that shell, and m
// counts up from -l -- the same order Orbital is declared in.
constexpr std::optional<Orbital> orbitalFor(i32 n, i32 l, i32 m) {
    constexpr i32 kMaxN = 4;
    if (n < 1 || n > kMaxN || l < 0 || l >= n || m < -l || m > l) {
        return std::nullopt;
    }
    const i32 shell_start = (n - 1) * n * (2 * n - 1) / 6;
    return static_cast<Orbital>(shell_start + l * l + (m + l));
}

struct QuantumNumbers {
    i32 n;
    i32 l;
    i32 m;
};

// The inverse of orbitalFor: the |n l m> an orbital stands for. {0, 0, 0} for Orbital::Count.
constexpr QuantumNumbers quantumNumbersOf(Orbital orbital) {
    for (i32 n = 1; n <= 4; n++) {
        for (i32 l = 0; l < n; l++) {
            for (i32 m = -l; m <= l; m++) {
                if (orbitalFor(n, l, m) == orbital) {
                    return QuantumNumbers{n, l, m};
                }
            }
        }
    }
    return QuantumNumbers{0, 0, 0};
}

// A short ASCII label for UI and logs ("1s", "2p_x", "3d_x2-y2", "4f_y(3x2-y2)").
constexpr const char* orbitalName(Orbital orbital) {
    constexpr const char* kNames[] = {
        "1s",
        "2s", "2p_y", "2p_z", "2p_x",
        "3s", "3p_y", "3p_z", "3p_x", "3d_xy", "3d_yz", "3d_z2", "3d_xz", "3d_x2-y2",
        "4s", "4p_y", "4p_z", "4p_x", "4d_xy", "4d_yz", "4d_z2", "4d_xz", "4d_x2-y2",
        "4f_y(3x2-y2)", "4f_xyz", "4f_yz2", "4f_z3", "4f_xz2", "4f_z(x2-y2)", "4f_x(x2-3y2)",
    };
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == static_cast<u32>(Orbital::Count));
    const u32 index = static_cast<u32>(orbital);
    return index < static_cast<u32>(Orbital::Count) ? kNames[index] : "invalid";
}

static_assert(static_cast<u32>(Orbital::Count) == 30, "Orbital must list every orbital up to n = 4");
static_assert(orbitalFor(1, 0, 0) == Orbital::S1);
static_assert(orbitalFor(2, 1, -1) == Orbital::P2y);
static_assert(orbitalFor(2, 1, 1) == Orbital::P2x);
static_assert(orbitalFor(3, 0, 0) == Orbital::S3);
static_assert(orbitalFor(3, 2, 0) == Orbital::D3z2);
static_assert(orbitalFor(3, 2, 2) == Orbital::D3x2_y2);
static_assert(orbitalFor(4, 0, 0) == Orbital::S4);
static_assert(orbitalFor(4, 2, -2) == Orbital::D4xy);
static_assert(orbitalFor(4, 3, -3) == Orbital::F4y_3x2_y2);
static_assert(orbitalFor(4, 3, -1) == Orbital::F4yz2);
static_assert(orbitalFor(4, 3, 1) == Orbital::F4xz2);
static_assert(orbitalFor(4, 3, 3) == Orbital::F4x_x2_3y2);
static_assert(!orbitalFor(0, 0, 0) && !orbitalFor(2, 2, 0) && !orbitalFor(2, 1, 2) && !orbitalFor(5, 0, 0));
static_assert(quantumNumbersOf(Orbital::P3x).n == 3 && quantumNumbersOf(Orbital::P3x).l == 1 && quantumNumbersOf(Orbital::P3x).m == 1);
static_assert(quantumNumbersOf(Orbital::F4y_3x2_y2).m == -3);

} // namespace renderer
