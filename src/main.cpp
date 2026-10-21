#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <app/application.hpp>
#include <graphics/presenter.hpp>
#include <graphics/renderers/clear_renderer.hpp>

#include <sokol_gfx.h>

// First light: a window cleared to solid yellow through GameEngine's ClearRenderer and
// Presenter. Proves the engine submodule builds, the window opens, and frames present.
// The electron-cloud renderer replaces ClearRenderer here.

namespace {

constexpr sg_color kYellow = {1.0f, 1.0f, 0.0f, 1.0f};

// The app's State. app::Application builds it once the GPU is up and destroys it before the
// GPU goes down, so members holding GPU handles are plain members.
struct ElectronCloud {
        graphics::renderers::ClearRenderer renderer{kYellow};
        graphics::Presenter presenter;

        explicit ElectronCloud(app::Context&) {}

        void frame(app::Context&) { presenter.present(renderer); }
};

} // namespace

int main(int argc, char** argv) {
    app::Application<ElectronCloud> app{{.title = "ElectronCloud"}, argc, argv};
    app.run();

    return 0;
}
