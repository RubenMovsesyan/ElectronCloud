#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <app/application.hpp>
#include <graphics/presenter.hpp>
#include <renderer/electron_cloud_renderer.hpp>

// ElectronCloud: our own renderer (src/renderer/) plugged into GameEngine's Presenter.

namespace {

// The app's State. app::Application builds it once the GPU is up and destroys it before the
// GPU goes down, so members holding GPU handles are plain members.
struct ElectronCloud {
        renderer::ElectronCloudRenderer renderer;
        graphics::Presenter presenter;

        explicit ElectronCloud(app::Context&) {}

        void frame(app::Context&) { presenter.present(renderer); }
};

} // namespace

int main(int argc, char** argv) {
    initLog(1024 * 1024);
    RLOG(LL_INFO, "Starting ElectronCloud");

    app::Application<ElectronCloud> app{{.width = 1920, .height = 1080, .title = "ElectronCloud", .resizable = false}, argc, argv};
    app.run();

    return 0;
}
