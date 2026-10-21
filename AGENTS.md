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
docs/                 HTML docs (dark mode), docs/index.html is the overview
```

## Building

Needs what GameEngine needs: `vulkan-headers`, `vulkan-validation-layers`, and X11
development headers (`libx11`, `libxi`, `libxcursor`).

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

## Working on GameEngine

GameEngine doesn't yet support everything ElectronCloud needs, so it is developed here too.

- **Engine features go in the engine, not here.** A renderer, a pass, a scene type -- if it
  is reusable, write it in `external/GameEngine/src/` following
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
- No CUDA here (yet): `cuda_interop.cpp` compiles to nothing without `GAMEENGINE_CUDA`.
