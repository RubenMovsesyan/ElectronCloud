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
// Convention: m -> name follows the project's table as given. In the usual real-orbital
// convention, m < 0 is the sin(|m| phi) combination and m > 0 the cos one. The d row follows
// it (m -1 = d_yz, m +1 = d_xz) and so do f at m = +/-2, but p and f at odd m are swapped
// relative to it (the usual convention has p: m -1 = p_y, m +1 = p_x; f: m -3 = y(3x^2-y^2),
// m -1 = yz^2, m +1 = xz^2, m +3 = x(x^2-3y^2)). Harmless while the enum is only a name;
// settle it before the wave function picks an angular formula from it.
enum class Orbital : u32 {
    // n = 1
    S1, // 1s       (l 0, m 0)

    // n = 2
    S2,  // 2s      (l 0, m 0)
    P2x, // 2p_x    (l 1, m -1)
    P2z, // 2p_z    (l 1, m  0)
    P2y, // 2p_y    (l 1, m +1)

    // n = 3
    S3,      // 3s           (l 0, m 0)
    P3x,     // 3p_x         (l 1, m -1)
    P3z,     // 3p_z         (l 1, m  0)
    P3y,     // 3p_y         (l 1, m +1)
    D3xy,    // 3d_xy        (l 2, m -2)
    D3yz,    // 3d_yz        (l 2, m -1)
    D3z2,    // 3d_z^2       (l 2, m  0)
    D3xz,    // 3d_xz        (l 2, m +1)
    D3x2_y2, // 3d_x^2-y^2   (l 2, m +2)

    // n = 4
    S4,         // 4s               (l 0, m 0)
    P4x,        // 4p_x             (l 1, m -1)
    P4z,        // 4p_z             (l 1, m  0)
    P4y,        // 4p_y             (l 1, m +1)
    D4xy,       // 4d_xy            (l 2, m -2)
    D4yz,       // 4d_yz            (l 2, m -1)
    D4z2,       // 4d_z^2           (l 2, m  0)
    D4xz,       // 4d_xz            (l 2, m +1)
    D4x2_y2,    // 4d_x^2-y^2       (l 2, m +2)
    F4x_x2_3y2, // 4f_x(x^2-3y^2)   (l 3, m -3)
    F4xyz,      // 4f_xyz           (l 3, m -2)
    F4xz2,      // 4f_xz^2          (l 3, m -1)
    F4z3,       // 4f_z^3           (l 3, m  0)
    F4yz2,      // 4f_yz^2          (l 3, m +1)
    F4z_x2_y2,  // 4f_z(x^2-y^2)    (l 3, m +2)
    F4y_3x2_y2, // 4f_y(3x^2-y^2)   (l 3, m +3)

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

static_assert(static_cast<u32>(Orbital::Count) == 30, "Orbital must list every orbital up to n = 4");
static_assert(orbitalFor(1, 0, 0) == Orbital::S1);
static_assert(orbitalFor(2, 1, -1) == Orbital::P2x);
static_assert(orbitalFor(2, 1, 1) == Orbital::P2y);
static_assert(orbitalFor(3, 0, 0) == Orbital::S3);
static_assert(orbitalFor(3, 2, 0) == Orbital::D3z2);
static_assert(orbitalFor(3, 2, 2) == Orbital::D3x2_y2);
static_assert(orbitalFor(4, 0, 0) == Orbital::S4);
static_assert(orbitalFor(4, 2, -2) == Orbital::D4xy);
static_assert(orbitalFor(4, 3, -3) == Orbital::F4x_x2_3y2);
static_assert(orbitalFor(4, 3, 3) == Orbital::F4y_3x2_y2);
static_assert(!orbitalFor(0, 0, 0) && !orbitalFor(2, 2, 0) && !orbitalFor(2, 1, 2) && !orbitalFor(5, 0, 0));

} // namespace renderer
