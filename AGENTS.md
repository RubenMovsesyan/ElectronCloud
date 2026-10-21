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
  renderer/           ElectronCloudRenderer (C++ + its CUDA kernels in .cu files)
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

- CUDA runs **inside the renderer, on the render thread**. The constructor makes a
  `gpu::CudaInterop` (a CUDA context on the GPU sokol's Vulkan device uses, current on the
  calling thread) and loads `electron_cloud_renderer.ptx`. No worker thread.
- CUDA can only share **buffers** with sokol/Vulkan, not images: the engine's sokol fork
  exports buffer memory (`gpu::ExternalBuffer`, vertex + storage usage) and nothing else.
  So CUDA cannot write a texture directly; pixels have to go through a buffer.
- The pixel path is not built yet: `render()` clears to yellow (an internal
  `ClearRenderer`), and the constructor runs a smoke-test kernel and logs the result.

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
  `sg_view render(int width, int height)` handed to `graphics::Presenter`. The electron
  renderer will be one such type; today `src/main.cpp` uses
  `graphics::renderers::ClearRenderer` (a solid yellow window) as first light.

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
- The window is 1920x1080 with `Config::resizable = false`. On Hyprland that makes it
  float at that size; it is a window-manager hint, not a lock.
