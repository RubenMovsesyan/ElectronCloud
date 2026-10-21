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
each file in `cuda_kernels` with `nvcc --ptx -arch=sm_89` to `.build/cuda/<stem>.ptx`. A new
`.cu` file means a new line in `cuda_kernels`. The renderer loads its PTX at runtime with
`cuModuleLoad` from a path relative to the project root, so run from there (`./run` does).

## The renderer

`renderer::ElectronCloudRenderer` (`src/renderer/`) is deliberately **not** in GameEngine:
it is specific to this project. It plugs into the engine through the `graphics::Renderer`
concept (`sg_view render(int width, int height)`) and is drawn by `graphics::Presenter`
like any engine renderer.

It is a CUDA ray marcher. The State (`ElectronCloud` in `src/main.cpp`) owns the
`Camera` (fly controls, same as the engine sandboxes), a `renderer::Sphere`, and the
`RaymarchSettings` (`step` = delta t, `max_distance`); each frame it calls
`renderer.beginFrame(camera)`, `renderer.submit(sphere)`, then `presenter.present(renderer)`.

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
- The kernel: one thread per pixel; a ray from the camera advances `step` at a time out to
  `max_distance`. At each step `checkRay()` returns the cloud's density there (an `f32`;
  currently the test sphere: 1 at the centre, linear to 0 at the surface). Each sample is
  turned into an opacity with Beer-Lambert (`alpha = 1 - exp(-density * step *
  kAbsorption)`, `kAbsorption = 10`) and blended front to back with colour
  `(0.3, 0.6, 1.0) * density`; the ray stops early at 95% opacity, and leftover light shows
  the background. Samples at or below 0.001 density are skipped.
- Kernel code uses `f32` from `common.h` (nvcc gets `-Iexternal/devtools`), not `float`;
  CUDA's own vector types (`float3`) stay as they are.
- `raymarch_params.h` is plain C data (`f32`s, unsigneds) passed by value to
  `cuLaunchKernel`; keep it free of Eigen and CUDA vector types so both compilers agree on
  its layout. Headers the kernels include are listed in `cuda_headers` in `build.cpp` so a
  change to them rebuilds the PTX.
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
