#include "buildcpp/build.hpp"

#include <functional>
#include <vector>

// ElectronCloud builds GameEngine from source (external/GameEngine, a submodule) and links
// it straight into one binary -- no separate engine library. GameEngine's own build.cpp is
// not run; the engine sources and include paths below mirror it, so when the engine gains
// a source file, add it to engine_sources here too.

const std::string BUILD_DIR = ".build";
const std::string ENGINE = "external/GameEngine";

int main(int argc, char** argv) {
    initLog(4096);
    Build build(BUILD_DIR, "clang++", argc, argv);

    // mkdir has no staleness of its own to track -- idempotent, just needs to run before
    // the shader Task below.
    Command mkdir_shaders({"mkdir", "-p", BUILD_DIR + "/shaders"});
    mkdir_shaders.exec();

    BuildGroup& app = build.addGroup();
    app.addCompileFlag("-std=c++23");
    app.addLinkFlag("-std=c++23");

    app.addInclude(Include<Direct>("src"));
    app.addInclude(Include<Direct>("external/devtools"));
    app.addInclude(Include<Direct>(ENGINE + "/src"));
    app.addInclude(Include<Direct>(ENGINE + "/external/sokol"));
    app.addInclude(Include<Direct>(ENGINE + "/external/sokol/util"));
    app.addInclude(Include<Direct>(ENGINE + "/external/imgui"));
    app.addInclude(Include<Direct>(ENGINE + "/external/Eigen"));
    app.addInclude(Include<Direct>(ENGINE + "/external/cgltf"));
    app.addInclude(Include<Direct>(ENGINE + "/external/stb"));
    app.addInclude(Include<Direct>(BUILD_DIR + "/shaders"));

    app.addLink(Link<Dependency>("vulkan"));
    app.addLink(Link<Dependency>("pthread"));

    // sokol_app on Linux: X11 windowing plus dlopen for the Vulkan loader.
    app.addLink(Link<Dependency>("X11"));
    app.addLink(Link<Dependency>("Xi"));
    app.addLink(Link<Dependency>("Xcursor"));
    app.addLink(Link<Dependency>("dl"));
    app.addLink(Link<Dependency>("m"));

    // The engine's shaders -> one sokol-shdc generated header carrying the SPIR-V for every
    // program. A Command Output is always considered stale, so only add the Task when the
    // header is missing or older than its source.
    const std::string shader_source = ENGINE + "/src/graphics/shaders/engine.glsl";
    const std::string shader_header = BUILD_DIR + "/shaders/engine.glsl.h";
    const bool shader_stale = !std::filesystem::exists(shader_header) ||
                              std::filesystem::last_write_time(shader_source) > std::filesystem::last_write_time(shader_header);

    Task* shader_task = nullptr;
    if (shader_stale) {
        shader_task = &app.addTask(Output(Command({ENGINE + "/external/sokol-tools-bin/bin/linux/sokol-shdc", "-i", shader_source,
                                                   "-o", shader_header, "-l", "spirv_vk", "-f", "sokol"})));
    }

    // Mirror of engine_sources in external/GameEngine/build.cpp. No CUDA: cuda_interop.cpp
    // is fully #ifdef GAMEENGINE_CUDA and compiles to nothing.
    const std::vector<std::string> engine_sources = {
        "src/gpu/sokol.cpp",
        "src/gpu/handle.cpp",
        "src/gpu/external_buffer.cpp",
        "src/gpu/cross_semaphore.cpp",
        "src/gpu/cuda_interop.cpp",
        "src/gpu/shared_instance_buffer.cpp",

        "external/imgui/imgui.cpp",
        "external/imgui/imgui_draw.cpp",
        "external/imgui/imgui_tables.cpp",
        "external/imgui/imgui_widgets.cpp",

        "src/graphics/primitives/scene/camera.cpp",
        "src/graphics/primitives/scene/frustum.cpp",
        "src/graphics/primitives/scene/light.cpp",
        "src/graphics/primitives/scene/mesh.cpp",
        "src/graphics/primitives/scene/texture.cpp",
        "src/graphics/primitives/scene/material.cpp",

        "src/graphics/primitives/pass/shadow.cpp",
        "src/graphics/primitives/pass/light_manager.cpp",
        "src/graphics/primitives/pass/debug_draw.cpp",
        "src/graphics/primitives/pass/render_target.cpp",
        "src/graphics/primitives/pass/fullscreen_pass.cpp",
        "src/graphics/primitives/pass/post.cpp",

        "src/graphics/renderers/deferred_renderer.cpp",
        "src/graphics/renderers/point_renderer.cpp",
        "src/graphics/renderers/clear_renderer.cpp",

        "src/graphics/presenter.cpp",

        "src/scene/model.cpp",

        "src/assets/asset_library.cpp",
        "src/assets/loaders/defaults.cpp",
        "src/assets/loaders/gltf.cpp",
        "src/assets/loaders/image.cpp",

        "src/app/context.cpp",
        "src/app/cli_parser.cpp",
        "src/app/input_state.cpp",
        "src/app/debug_ui.cpp",
    };

    // ElectronCloud's own sources. Adding a file means adding a line here -- no globbing.
    const std::vector<std::string> sources = {
        "src/main.cpp",
    };

    std::vector<std::reference_wrapper<Task>> objects;
    for (const auto& source : engine_sources) {
        Task& task = app.addTask(Output(Object(ENGINE + "/" + source)));
        // Only orders the compile after sokol-shdc; rebuilding on a shader change comes from
        // the include scan of engine.glsl.h, so ordering every engine TU costs nothing.
        if (shader_task != nullptr) {
            task.depends_on(*shader_task);
        }
        objects.push_back(task);
    }
    for (const auto& source : sources) {
        objects.push_back(app.addTask(Output(Object(source))));
    }

    Task& exe = app.addTask(Output(Binary("electron_cloud")));
    for (Task& object : objects) {
        exe.depends_on(object);
    }

    build.build();

    return 0;
}
