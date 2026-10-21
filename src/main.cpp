#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <app/application.hpp>
#include <graphics/presenter.hpp>
#include <graphics/renderers/clear_renderer.hpp>

#include <sokol_gfx.h>

#include <optional>

// First light: a window cleared to solid yellow through GameEngine's ClearRenderer and
// Presenter. Proves the engine submodule builds, the window opens, and frames present.
// The electron-cloud renderer replaces ClearRenderer here.

namespace {

constexpr sg_color kYellow = {1.0f, 1.0f, 0.0f, 1.0f};

struct State {
        // optional: both own GPU handles, so they are built after sg_setup() (init) and
        // released before sg_shutdown() (cleanup).
        std::optional<graphics::renderers::ClearRenderer> renderer;
        std::optional<graphics::Presenter> presenter;
};

} // namespace

int main() {
    State state;

    app::Application app{
        {.width = 1280, .height = 720, .title = "ElectronCloud"},
        [&](app::Application&) {
            state.renderer.emplace(kYellow);
            state.presenter.emplace();
        },
        [&](app::Application&) { state.presenter->present(*state.renderer); },
        [&](app::Application&) {
            state.presenter.reset();
            state.renderer.reset();
        },
    };
    app.run();

    return 0;
}
