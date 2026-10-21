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
  renderer/           ElectronCloudRenderer: C++, its CUDA kernels (.cu), the params struct
                      both share (raymarch_params.h), and shaders/ (sokol-shdc GLSL)
docs/                 HTML docs (dark mode), docs/index.html is the overview
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
```

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
`renderer::Atom`, and a simulation clock; each frame it calls
`renderer.beginFrame(camera, simulation_time)`, `renderer.submit(atom)`, then
`presenter.present(renderer, overlay)`.

What the renderer is given -- one set of types in `raymarch_params.h`, shared by `main.cpp`,
the C++ renderer and the kernel (no mirror structs, no conversion code):

- `Atom`: everything the density depends on. Nucleus position, nuclear charge
  `Z`, `bohr_radius` (world units per Bohr radius -- the cloud's size on screen),
  `bounds_radius` (density is 0 outside; rays are only marched inside it), and up to two
  `QuantumState`s `|n l m>` with a complex weight `amplitude * e^(i * phase)`. Two states
  model a transition between energy levels as a superposition whose amplitudes the app
  moves over time.
- Simulation time: `f64`, in atomic units of time (hbar / E_h ~= 2.419e-17 s), advanced
  by the State at `kAtomicTimePerSecond` per real second (10 today; real time would be
  ~4.1e16). `f64` so the phases stay exact as it grows.
- `RaymarchSettings`: `step_size`, `max_distance`, `min_density`,
  `opacity_cutoff` -- quality against speed.
- `Appearance`: cloud colour, background colour, absorption -- purely
  cosmetic.

`renderer::Orbital` (`src/renderer/orbital.hpp`, host-side only so far) names each spatial
orbital up to n = 4 (`S1`, `P2x`, `D3x2_y2`, `F4xyz`, ...), and `orbitalFor(n, l, m)` maps
quantum numbers to it (`nullopt` if invalid or n > 4). It is `constexpr`, and the enum is
declared in n, l, m order, which the lookup's index formula relies on. m -> real orbital
follows the usual convention (m < 0 = sin(|m| phi), m > 0 = cos(|m| phi): m -1 is p_y,
m +1 is p_x, and so on -- see the header); the wave function must use the same one.

The wave function is not in yet: `checkRay()` takes the atom and the time but still returns
a hard-coded test sphere (centre 0, radius 1, density 1 at the centre falling linearly to 0).

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
  sample `checkRay(position, atom, time)` returns the density (`f32`). Each sample becomes
  an opacity with Beer-Lambert (`alpha = 1 - exp(-density * step_size * absorption)`) and
  is blended front to back with `cloud_color * density`; the ray stops early at
  `opacity_cutoff`, samples at or below `min_density` are skipped, and leftover light shows
  `background_color`.
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
- An FPS counter (ImGui, through the engine's `DebugUI`) is drawn in the top-left corner as
  the Presenter's overlay. `DebugUI` starts hidden, so the State's constructor calls
  `setVisible(true)`. The number is ImGui's 60-frame average; with vsync it caps at the
  monitor's refresh rate.
- The window is 1920x1080 with `Config::resizable = false`. On Hyprland that makes it
  float at that size; it is a window-manager hint, not a lock.
