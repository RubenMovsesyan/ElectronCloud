# ElectronCloud

An electron cloud renderer -- what an electron "looks like". Built on GameEngine, which is
developed alongside it as a submodule.

## Layout

```
buildcpp/             git submodule, the build system (build.hpp)
build.cpp             the build script, C++23
external/             one dependency per directory, each a git submodule
  devtools/           header-only C utilities: rlog.h, common.h (u32, f32, ...), attr.h
  GameEngine/         the engine, cloned from its remote, with its own submodules (--recursive)
src/                  ElectronCloud's own code; src/main.cpp is the entry point
  animation/          keyframe system for scripted recordings (Track, Timeline)
  renderer/           ElectronCloudRenderer: C++, its CUDA kernels (.cu), the params struct
                      both share (raymarch_params.h), and shaders/ (sokol-shdc GLSL)
docs/                 HTML docs (dark mode), docs/index.html is the overview
  media/              example recordings (.mp4) and their README GIF previews
README.md             the GitHub front page: example videos, what it draws, how to run it
```

## Building

Needs what GameEngine needs: `vulkan-headers`, `vulkan-validation-layers`, and X11
development headers (`libx11`, `libxi`, `libxcursor`). Plus the **CUDA toolkit**, which is
required: `build.cpp` looks in `CUDA_PATH`, `/opt/cuda`, `/usr/local/cuda` and stops if it
finds none.

```sh
git submodule update --init --recursive   # after cloning
./run                                     # bootstrap ./build if needed, build, launch
./run clean                               # wipe .build first
./run --record out.mp4 [--from 1s] [--to 2p_z] [--elevation 20] [--intensity 1e12] \
      [--fps 60] [--width 1920] [--height 1080]   # record mode (needs ffmpeg)
```

## Modes

`src/main.cpp` runs in one of two modes, picked in the State's constructor from the command
line:

- **Live** (default): `graphics::Presenter`, wall-clock time, fly camera, the debug panel.
- **Record** (`--record <file>`): `graphics::Recorder` (GameEngine) renders each frame at the
  video size, shows a preview in the window and writes the frame to the video through
  ffmpeg (fatal at startup if ffmpeg is missing). Everything advances by exactly
  `recorder->frameTime()` (1 / fps) per frame and the script's time is
  `frame index / fps`, so the video is deterministic and plays in real time however long
  frames take to render. The ImGui overlay (FPS + a progress panel) is on the window only,
  never in the video. When the script ends the Recorder is finished and the app quits.

The script is hard-coded in `ElectronCloud::recordingScript()` with the keyframe system in
`src/animation/timeline.hpp`: `Track<T>` (values eased between keys, held outside them) and
`Timeline` (cues fired as time passes, plus an end time). Today: 21 s -- orbit the `--from`
orbital for 4 s, at 4 s a 10 s transition to `--to` (the cue calls `startTransition`), then
7 more seconds on it; the durations are named constants in `recordingScript()`. The camera
orbits at 0.8 rad/s at `--elevation` degrees (default 20; clamped to +-89 because the camera
basis comes from world up -- 89 is a top view) and its distance eases between the two shells'
framing distances (`cameraDistanceFor`: 6 / 16 / 26 A for n = 1 / 2 / 3) across the
transition.

Record-mode options:
- `--from` / `--to`: orbital names as in the menus (`1s`, `2p_z`, `3d_x2-y2`, ...;
  `orbitalFromName`), default 1s -> 2p_z. Unknown names or from == to are fatal; forbidden
  pairs record with a log note.
- `--intensity` (W/cm^2, default 1e12): the physical knob for how much the cloud sloshes,
  ~1 / sqrt(intensity). The log prints the slosh count and real duration. 2p_z -> 3d_z2:
  2.6 sloshes at 1e12, 20 at 1.75e10. Past ~150 sloshes in a 10 s transition (fewer than ~4
  frames each) they alias into flicker. A playing transition advances before the cues fire each frame, so a
transition started on a frame shows its first moment, and the end-of-transition check has a
1e-9 tolerance because elapsed time is a sum of 1 / fps steps.

`./build` rebuilds itself when `build.cpp` or `buildcpp/build.hpp` changes and writes
`compile_commands.json`. The binary is `.build/bin/electron_cloud`.

Shaders: `build.cpp` runs the engine's bundled `sokol-shdc` over the engine's
`engine.glsl` and over our `src/renderer/shaders/electron_cloud.glsl`, generating
`.build/shaders/<name>.glsl.h`. A storage buffer in sokol-shdc GLSL must be an array of a
struct (`ec_pixel pixels[]`, not `uint pixels[]`).

`build.cpp` compiles GameEngine **from source** straight into our binary -- the engine's
own `build.cpp` is never run. Its `engine_sources` list mirrors the one in
`external/GameEngine/build.cpp`, and the engine's shader header is generated with the
engine's bundled `sokol-shdc` into `.build/shaders/`. Adding a source file means adding a
line to `sources` (ours) or `engine_sources` (engine) -- there is no globbing.

CUDA: `build.cpp` defines `GAMEENGINE_CUDA` for every source (ours and the engine's, which
switches on `src/gpu/cuda_interop.*`), links the Driver API (`libcuda`) only, and compiles
each file in `cuda_kernels` to `.build/cuda/<stem>.ptx` with **clang's own CUDA support**
(`clang++ -x cuda --cuda-gpu-arch=sm_89 --cuda-device-only -S`) -- nvcc is not used. clang
still needs the CUDA toolkit's headers and libdevice (the device math library behind
`expf`, `sqrtf`, ...), found through `--cuda-path`. A new `.cu` file means a new line in
`cuda_kernels`.

Why clang, not nvcc: one compiler for host and device code (same C++23, same struct
layout rules); `constexpr` functions are usable on the device without nvcc's
`--expt-relaxed-constexpr`, so Eigen and `std::array` work in kernels with no flags or
warnings; and on the raymarch kernel it was ~15-20% faster than nvcc's PTX. The cost:
clang lags new CUDA releases. It officially knows CUDA up to 13.2 while the installed
toolkit is 13.4, so `-Wno-unknown-cuda-version` hides that warning. If a toolkit upgrade
breaks the kernel build, check that first. The renderer loads its PTX at runtime with
`cuModuleLoad` from a path relative to the project root, so run from there (`./run` does).

## The renderer

`renderer::ElectronCloudRenderer` (`src/renderer/`) is deliberately **not** in GameEngine:
it is specific to this project. It plugs into the engine through the `graphics::Renderer`
concept (`sg_view render(int width, int height)`) and is drawn by `graphics::Presenter`
like any engine renderer.

It is a CUDA ray marcher through an electron cloud. The State (`ElectronCloud` in
`src/main.cpp`) owns the `Camera` (fly controls, same as the engine sandboxes), the
`renderer::Atom`, and the transition that may be playing; each frame it calls
`renderer.beginFrame(camera, simulation_time)`, `renderer.submit(atom)`, then
`presenter.present(renderer, overlay)`.

What the renderer is given -- one set of types in `raymarch_params.h`, shared by `main.cpp`,
the C++ renderer and the kernel (no mirror structs, no conversion code):

- `Atom`: everything the density depends on. Nucleus position, nuclear charge
  `Z`, `bohr_radius` (world units per Bohr radius -- the cloud's size on screen),
  `bounds_radius` (density is 0 outside; rays are only marched inside it), and up to two
  `QuantumState`s `|n l m>`. Two states (`state_count = 2`) are a transition in progress:
  a superposition whose density is `a psi_A^2 + b psi_B^2 + interference psi_A psi_B`, with
  the three `Superposition` weights worked out by the CPU for the current moment.
- Simulation time: `f64`, in atomic units of time (hbar / E_h ~= 2.419e-17 s), measured
  from the start of the current transition (0 when none is playing). The kernel doesn't use
  it -- the weights already encode it -- but it is passed along.
- `RaymarchSettings`: `step_size`, `max_distance`, `min_density`,
  `opacity_cutoff` -- quality against speed.
- `Appearance`: cloud colour, background colour, absorption, brightness -- purely
  cosmetic.

`renderer::Orbital` (`src/renderer/orbital.hpp`, host-side only so far) names each spatial
orbital up to n = 4 (`S1`, `P2x`, `D3x2_y2`, `F4xyz`, ...), and `orbitalFor(n, l, m)` maps
quantum numbers to it (`nullopt` if invalid or n > 4). It is `constexpr`, and the enum is
declared in n, l, m order, which the lookup's index formula relies on. m -> real orbital
follows the usual convention (m < 0 = sin(|m| phi), m > 0 = cos(|m| phi): m -1 is p_y,
m +1 is p_x, and so on -- see the header); the wave function must use the same one.

The wave function, so far: `checkRay(position, atom, time)` converts the sample to
spherical coordinates around the nucleus, looks up the orbital of `atom.states[0]` with
`orbitalFor`, and returns `psi * psi` from `waveFunction(orbital, point, a)`
(`src/renderer/hydrogen.hpp`, `a = bohr_radius / nuclear_charge`). Implemented for every
orbital up to n = 3 (1s .. 3d); n = 4 and invalid quantum numbers are density 0. With two
states it evaluates both wave functions and returns `a psi_A^2 + b psi_B^2 + interference
psi_A psi_B` (clamped at 0), so a transition costs ~2x a single orbital.

Transitions (`src/renderer/transition.hpp`, CPU-only, f64, atomic units): resonant light
polarised along the transition dipole `d` drives A -> B (Rabi oscillation):
`c_A = cos(Omega t/2)`, `c_B = -i sin(Omega t/2)`, `Omega = |d| E0`. With `c_B` imaginary the
interference term is `-sin(Omega t) sin(omega0 t)`, `omega0 = E_B - E_A` (`E_n = -Z^2/(2n^2)`)
-- not `2 c_A c_B cos(omega0 t)`, which only holds for real amplitudes. The transition ends
at `t = pi / Omega` with everything in B; the cloud sloshes `|omega0| / (2 Omega)` times on
the way.

- Everything is physical except playback speed: the laser intensity (W/cm^2, debug panel,
  default 1e12) gives `E0`, the dipole gives `Omega`, so the real duration and the number of
  sloshes follow (1s -> 2p_z at 1e12 W/cm^2: 19 fs, 47 sloshes; ~1/sqrt(intensity)). The
  chosen playback seconds only map that real time onto the screen.
- `DipoleTable` integrates `<A|r|B>` numerically from `hydrogen.hpp` once (~0.2 s, built in
  the State's constructor) for every pair with `l` differing by 1 (all others are 0 by the
  angular-momentum rule). It matches the textbook values (`<1s|z|2p_z>` = 0.745,
  `<2s|z|2p_z>` = 3.00, `<2p_z|z|3d_z2>` = 2.45). Allowed = non-zero dipole; this gets
  real-orbital cases right that `|Delta m| <= 1` on the m labels doesn't (2p_x -> 3d_xy
  allowed, 2p_z -> 3d_x^2-y^2 forbidden).
- Forbidden transitions (dipole 0) would never happen with light; they are animated as if
  the dipole were `kForbiddenStandInDipole` (1 e a0) and flagged in the UI. Same-shell
  pairs (e.g. 2s -> 2p) have `omega0 = 0`: they transition but don't slosh ("dE = 0").
- Density scale during a transition blends the two peaks in **log space**
  (`peak_A^a * peak_B^b`): exact at both ends, one scale per frame. A linear blend stays
  pinned to the bigger peak (1s is ~60x 2p), so B stayed invisible and popped in at the end.
- At the end the State switches to B alone, which is exactly what the superposition is at
  `t = pi / Omega` -- the last transition frame and the first single-state frame match
  pixel for pixel. Bounds during a transition are the larger of the two shells'.
- Too many sloshes for the playback time alias into flicker; the panel warns when there are
  fewer than ~4 frames per slosh.

Colours (`src/renderer/orbital_color.hpp`, cosmetic, host-side): every orbital with a wave
function gets a hue spaced evenly round the wheel in `Orbital` order (1s red ... 3d_x^2-y^2
pink; HSV saturation 0.7, value 1). The State sets `appearance.cloud_color` every frame
(`updateCloudColor`): the current orbital's colour, or during a transition
`blendOrbitalColors(from, to, b)` -- the hue turned the shorter way round the wheel by the
target's population, so colour and density arrive together (an RGB blend would go grey).
The orbital names in both dropdowns are drawn in their colours.

- `hydrogen.hpp` is `__host__ __device__` (`HYDROGEN_HD`), so the same code runs in the
  kernel and on the CPU -- that is how it was checked: every orbital integrates to 1,
  different orbitals are orthogonal, and GPU renders match a CPU march pixel for pixel.
  It returns psi, not |psi|^2, so superpositions can add wave functions before squaring.
- `Spherical` keeps r and the **sines and cosines** of theta and phi, computed from the
  Cartesian offset (cos theta = z / r, sin theta cos phi = x / r, ...). Calling acos /
  atan2 / sin / cos per sample made the 1s march ~3.7x slower; the formulas only ever need
  the sines and cosines (cos 2phi, sin 2phi via double-angle identities).
- 3d_xy / 3d_x^2-y^2 use 1 / (81 sqrt(2 pi a^3)). The often-printed 1 / (162 sqrt(pi a^3))
  integrates to 1/2.
- **Density is scaled so every orbital peaks at 1.** Absolute peaks differ a lot (|psi|^2 at
  a = 0.529: ~2.15 for 1s, 0.036 for 2p, 0.004 for 3d), which left everything but 1s
  invisible. `ElectronCloudRenderer::submit` computes `peakDensity(orbital, a)` (CPU,
  ~0.3 ms: the brightest direction on a sphere, then the brightest radius along it --
  every orbital is radial * angular) only when the orbital or `a` changes, and passes
  `RaymarchParams::density_scale = 1 / peak`. The kernel multiplies each sample by it, so
  `min_density`, emission and absorption all work on 0 .. 1 density; `checkRay` itself
  still returns the physical |psi|^2.
- At 0.1% of the peak, clouds reach ~1.8 A (n = 1), ~6.6 A (n = 2), ~13 A (n = 3);
  `boundsRadiusFor` in `main.cpp` uses 2.5 / 7.5 / 14. A ray also stops at `max_distance`
  from the camera, so whole n = 3 clouds need it >= the camera's distance + 14.
- Known look: multi-shell s orbitals (2s, 3s) show mostly as a dot. Their peak is a spike
  at the nucleus, so scaling by it leaves the outer shells faint.

World units are angstroms: `main.cpp` sets `bohr_radius = 0.529`. The 1s density peaks at
~2.15 at the nucleus and drops below `min_density` (0.001) at ~2.0 A, so `bounds_radius`
is 2.5.

- CUDA runs **inside the renderer, on the render thread** -- no worker. The constructor
  makes a `gpu::CudaInterop` (a CUDA context on the GPU sokol's Vulkan device uses) and
  loads `electron_cloud_renderer.ptx`.
- CUDA can only share **buffers** with Vulkan in this engine, not images (the sokol fork
  exports buffer memory only). So the `raymarch` kernel writes RGBA8 pixels into a
  `gpu::ExternalBuffer` (`pixels[y * width + x]`, y = 0 at the top, red in the low byte),
  and a fullscreen pass (`shaders/electron_cloud.glsl`, `ec_pixels`) copies it into the
  output `RenderTarget` that `render()` returns.
- Sync, all inside `render()`: two `gpu::CrossSemaphore` timelines. `render_done` is
  signalled at the start of frame N for frame N-1 (sokol's `sg_commit` submitted that
  frame's copy after the last `render()` returned); CUDA waits on it, marches, signals
  `cuda_done` = N; the graphics queue waits on `cuda_done` before the copy pass.
- The kernel: one thread per pixel. The ray is marched only between where it enters and
  leaves the atom's bounds sphere (and no further than `max_distance` from the camera),
  `step_size` at a time; the loop variable is `distance` along the ray, not time. At each
  sample `checkRay(position, atom, time)` returns the density (`f32`). Blending is
  emission-absorption, front to back, with the two **separate and both linear in density**:
  a sample adds `cloud_color * brightness * density * step_size`, dimmed by the
  transmittance of the cloud in front, and blocks `alpha = 1 - exp(-density * step_size *
  absorption)` of the light behind it. With `absorption = 0` a pixel is `brightness` times
  the total density along its ray (an X-ray of the probability); higher absorption adds
  depth cues. The ray stops early at `opacity_cutoff`, samples at or below `min_density`
  are skipped, and leftover light shows `background_color`.
- Why not the textbook `color * density * alpha`: alpha is itself ~density, so brightness
  went as density^2 and faint zones vanished, and absorption scaled the light along with the
  opacity, so lowering it made the cloud darker. Brightness is the exposure knob now; it is
  linear, so zone-to-zone ratios hold until a pixel clips at 1.
- Low absorption means rays rarely reach `opacity_cutoff`, so they march their whole bounds:
  n = 3 orbitals (14 A bounds) take ~10.5 ms per frame at step 0.02, n = 2 ~6.3 ms, 1s ~2 ms.
- Kernel code uses `f32` / `u32` / ... from `common.h`, not `float`, and Eigen
  (`Eigen::Vector3f`, `.dot()`, `.norm()`, `.normalized()`) for vector maths -- the same
  types as the host side.
- `raymarch_params.h` (namespace `renderer`) is passed by value to `cuLaunchKernel`
  (`RaymarchParams`), so host and device must lay it out identically. Stick to `common.h`
  scalars, `Eigen::Vector3f` (three packed `f32`, no alignment) and `std::array` of those.
  **No 16-byte-aligned Eigen types** (`Vector4f`, `Matrix4f`) -- they would move fields.
  `static_assert`s at the end of the header pin every struct's size (and the `f64 time`
  offset); the host and device passes both check them, so a layout disagreement fails the
  build. Update them when a field changes. Headers the kernels include are listed in
  `cuda_headers` in `build.cpp` so a change to them rebuilds the PTX.
- `CudaInterop` has no unmap: a resize leaks the old pixel buffer's CUDA import until the
  renderer is destroyed. The window is a fixed size, so this does not happen today.
- The destructor drains CUDA and the graphics queue, then releases CUDA *before* the
  Vulkan buffer and semaphores it imported.

## Working on GameEngine

GameEngine doesn't yet support everything ElectronCloud needs, so it is developed here too.

- **Engine features go in the engine, not here.** A pass, a scene type, interop
  plumbing -- if it is reusable, write it in `external/GameEngine/src/` following
  `external/GameEngine/AGENTS.md` (no virtual functions in `gpu/`/`graphics/`, a docs page
  per module, etc.). ElectronCloud's `src/` holds only what is specific to electrons.
- A new engine source file goes in **both** `external/GameEngine/build.cpp` and our
  `engine_sources`.
- Commit inside the submodule, push it to the engine's remote, then commit the moved
  submodule pointer here. A pointer to an unpushed engine commit breaks fresh clones.
- The engine renders through the `graphics::Renderer` concept: a type with
  `sg_view render(int width, int height)` handed to `graphics::Presenter`.
  `renderer::ElectronCloudRenderer` is one such type, kept outside the engine (see "The
  renderer").

## Gotchas

- `#define RLOG_IMPLEMENTATION` lives in `src/main.cpp`, exactly once.
- The app is the `ElectronCloud` State in `src/main.cpp`, run by
  `app::Application<ElectronCloud>`. It must satisfy `app::AppState`: an
  `explicit ElectronCloud(app::Context&)` (init) and `void frame(app::Context&)`; its
  destructor is cleanup. The Application builds it after `sg_setup()` and destroys it
  before `sg_shutdown()`, so GPU-owning members (renderers, the presenter) are plain
  members. Command-line options are parsed in the constructor from
  `context.argc()/argv()` with `app::CliParser`.
- `external/devtools` is ours and is the only devtools on the include path; GameEngine's
  own `external/GameEngine/external/devtools` copy is not used. Keep them on the same
  commit if the engine starts depending on newer devtools.
- `GAMEENGINE_CUDA` changes the layout of engine classes (`gpu::SharedInstanceBuffer`), and
  buildcpp does not rebuild on flag changes. After toggling it or any other compile flag,
  `./run clean`.
- A "Debug" ImGui panel below the FPS counter has an orbital dropdown listing every orbital
  with a wave function (`hasWaveFunction`); picking one sets `atom.states[0]` to its
  `|n l m>` (`quantumNumbersOf`) and `bounds_radius` to fit the shell
  (`boundsRadiusFor`: 2.5 / 7.5 / 14 A for n = 1 / 2 / 3). Below it, the transition
  controls: a target (every orbital, flagged "(forbidden)" or "(dE = 0)"), playback seconds,
  laser intensity, what the physics says about it (dipole, real duration, sloshes), and
  Start; while one plays, a progress bar, the populations and Cancel (back to the start
  orbital). The orbital dropdown is disabled during a transition. Then logarithmic Brightness
  and Absorption sliders edit the State's `appearance` and push it with `setAppearance`.
  Release mouse-look (ESC) to use it.
- An FPS counter (ImGui, through the engine's `DebugUI`) is drawn in the top-left corner as
  the Presenter's overlay. `DebugUI` starts hidden, so the State's constructor calls
  `setVisible(true)`. The number is ImGui's 60-frame average; with vsync it caps at the
  monitor's refresh rate.
- The window is 1920x1080 with `Config::resizable = false`. On Hyprland that makes it
  float at that size; it is a window-manager hint, not a lock.
