#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <app/application.hpp>
#include <graphics/presenter.hpp>
#include <graphics/primitives/scene/camera.hpp>
#include <renderer/electron_cloud_renderer.hpp>
#include <renderer/hydrogen.hpp>
#include <renderer/orbital.hpp>

#include <common.h>

#include <Eigen/Dense>

#include <imgui.h>

#include <sokol_app.h>

#include <algorithm>
#include <chrono>
#include <cmath>

// ElectronCloud: our own CUDA ray marcher (src/renderer/) plugged into GameEngine's Presenter,
// with a fly camera marking where the rays start, an FPS counter in the top-left corner and a
// debug panel to pick the orbital.
//
//   WASD move, Space / Left Ctrl up / down, ESC toggles mouse-look (release it to use the
//   debug panel).

namespace {

using graphics::primitives::scene::Camera;

constexpr f32 kMoveSpeed = 2.0f;        // units/sec -- the scene is a few units across
constexpr f32 kLookSensitivity = 0.15f; // radians per pixel of mouse delta, per second
constexpr f32 kPitchLimit = 1.5533f;    // ~89 degrees
constexpr f32 kFovY = 1.0472f;          // 60 degrees

// Atomic units of time (hbar / E_h ~= 2.419e-17 s) that pass per real second. Real time would
// be ~4.1e16; this slows the electron down enough to watch. As a feel for it: a 1s <-> 2p
// superposition in hydrogen oscillates with a period of ~16.8 atomic units.
constexpr f64 kAtomicTimePerSecond = 10.0;

const Eigen::Vector3f kWorldUp{0.0f, 1.0f, 0.0f};

f32 aspectOf(const app::Context& context) {
    return context.height() > 0 ? static_cast<f32>(context.width()) / static_cast<f32>(context.height()) : 1.0f;
}

// A small see-through panel pinned to the top-left corner. ImGui's Framerate is averaged over
// the last 60 frames, so the number is steady enough to read.
// How far out from the nucleus to march for an orbital of shell n, in angstroms (a0 = 0.529).
// The renderer scales density so each orbital peaks at 1; these cover where it stays above
// min_density (0.001, so 0.1% of the peak) -- ~1.8 A for n = 1, ~6.6 A for n = 2, ~13 A for
// n = 3 -- with some margin. A ray also stops at max_distance from the camera, so an n = 3
// cloud needs max_distance >= the camera's distance + 14 to be seen whole.
f32 boundsRadiusFor(i32 n) {
    switch (n) {
        case 1: return 2.5f;
        case 2: return 7.5f;
        default: return 14.0f;
    }
}

void drawFpsCounter() {
    constexpr f32 kMargin = 10.0f;
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoInputs;

    ImGui::SetNextWindowPos(ImVec2(kMargin, kMargin));
    ImGui::SetNextWindowBgAlpha(0.35f);
    if (ImGui::Begin("FPS", nullptr, kFlags)) {
        const f32 fps = ImGui::GetIO().Framerate;
        ImGui::Text("%.0f FPS  (%.2f ms)", fps, fps > 0.0f ? 1000.0f / fps : 0.0f);
    }
    ImGui::End();
}

// The app's State. app::Application builds it once the GPU is up and destroys it before the
// GPU goes down, so members holding GPU handles are plain members.
struct ElectronCloud {
        // Where the rays start. Starts 3 units back on +Z, looking at the origin.
        Camera camera;
        f32 yaw = 0.0f; // 0, 0 matches the starting direction (0, 0, -1)
        f32 pitch = 0.0f;

        // The atom whose electron cloud is drawn: hydrogen in its ground state, |1 0 0>, at the
        // origin. One world unit is one angstrom: a0 = 0.529 A.
        renderer::Atom atom{
            .nucleus = Eigen::Vector3f::Zero(),
            .nuclear_charge = 1.0f,
            .bohr_radius = 0.529f,
            // 1s density falls below min_density (0.001) at ~2.0 A; leave some margin.
            .bounds_radius = 2.5f,
            .states = {renderer::QuantumState{.n = 1, .l = 0, .m = 0, .amplitude = 1.0f, .phase = 0.0f}},
            .state_count = 1,
        };
        // Simulation time in atomic units, advanced by kAtomicTimePerSecond per real second.
        f64 simulation_time = 0.0;

        // How the cloud looks; brightness and absorption are tunable from the debug panel.
        renderer::Appearance appearance{
            .cloud_color = {0.3f, 0.6f, 1.0f},
            .background_color = {0.02f, 0.02f, 0.03f},
            .absorption = 0.5f,
            .brightness = 1.5f,
        };

        renderer::ElectronCloudRenderer renderer{
            renderer::RaymarchSettings{.step_size = 0.02f, .max_distance = 100.0f, .min_density = 0.001f, .opacity_cutoff = 0.95f},
            appearance,
        };
        graphics::Presenter presenter;

        std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();

        explicit ElectronCloud(app::Context& context)
            : camera(kFovY, aspectOf(context), 0.1f, 100.0f, Eigen::Vector3f{0.0f, 0.0f, 3.0f}, Eigen::Vector3f{0.0f, 0.0f, -1.0f}, kWorldUp) {
            // DebugUI starts hidden; the FPS counter is always on.
            context.debugUI().setVisible(true);
        }

        void frame(app::Context& context) {
            const auto now = std::chrono::steady_clock::now();
            const f32 dt = std::chrono::duration<f32>(now - last_time).count();
            last_time = now;

            if (context.input().isKeyPressed(SAPP_KEYCODE_ESCAPE)) {
                context.toggleCursorCapture();
            }
            camera.setAspectRatio(aspectOf(context));
            flyCamera(context, dt);
            simulation_time += static_cast<f64>(dt) * kAtomicTimePerSecond;

            renderer.beginFrame(camera, simulation_time);
            renderer.submit(atom);
            presenter.present(renderer, [&]() {
                context.debugUI().draw(context.width(), context.height(), dt, sapp_dpi_scale(), [&]() {
                    drawFpsCounter();
                    drawDebugPanel();
                });
            });
        }

        // Shows `selected` from now on: the atom's (only) state becomes its |n l m>, and the
        // bounds grow or shrink to fit the shell.
        void selectOrbital(renderer::Orbital selected) {
            const renderer::QuantumNumbers numbers = renderer::quantumNumbersOf(selected);
            atom.states[0].n = numbers.n;
            atom.states[0].l = numbers.l;
            atom.states[0].m = numbers.m;
            atom.state_count = 1;
            atom.bounds_radius = boundsRadiusFor(numbers.n);
        }

        // Below the FPS counter: an orbital dropdown, one at a time, listing only the orbitals
        // that have a wave function.
        void drawDebugPanel() {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 50.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                const auto& state = atom.states[0];
                const renderer::Orbital current = renderer::orbitalFor(state.n, state.l, state.m).value_or(renderer::Orbital::Count);
                if (ImGui::BeginCombo("Orbital", renderer::orbitalName(current))) {
                    for (u32 i = 0; i < static_cast<u32>(renderer::Orbital::Count); i++) {
                        const auto orbital = static_cast<renderer::Orbital>(i);
                        if (!renderer::hasWaveFunction(orbital)) {
                            continue;
                        }
                        const bool selected = orbital == current;
                        if (ImGui::Selectable(renderer::orbitalName(orbital), selected)) {
                            selectOrbital(orbital);
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                const renderer::QuantumNumbers numbers = renderer::quantumNumbersOf(current);
                ImGui::Text("n = %d, l = %d, m = %d", numbers.n, numbers.l, numbers.m);

                ImGui::Separator();
                // Brightness scales the whole image linearly; absorption only sets how much the
                // front of the cloud hides the back (0 = see-through).
                bool changed = ImGui::SliderFloat("Brightness", &appearance.brightness, 0.0f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                changed |= ImGui::SliderFloat("Absorption", &appearance.absorption, 0.0f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                if (changed) {
                    renderer.setAppearance(appearance);
                }
            }
            ImGui::End();
        }

        // Same controls as GameEngine's sandboxes.
        void flyCamera(app::Context& context, f32 dt) {
            const Eigen::Vector3f forward = camera.direction().normalized();
            const Eigen::Vector3f right = forward.cross(kWorldUp).normalized();
            const app::InputState& input = context.input();

            if (input.isKeyDown(SAPP_KEYCODE_W)) *camera.mutablePosition() += forward * kMoveSpeed * dt;
            if (input.isKeyDown(SAPP_KEYCODE_S)) *camera.mutablePosition() -= forward * kMoveSpeed * dt;
            if (input.isKeyDown(SAPP_KEYCODE_D)) *camera.mutablePosition() += right * kMoveSpeed * dt;
            if (input.isKeyDown(SAPP_KEYCODE_A)) *camera.mutablePosition() -= right * kMoveSpeed * dt;
            if (input.isKeyDown(SAPP_KEYCODE_SPACE)) *camera.mutablePosition() += kWorldUp * kMoveSpeed * dt;
            if (input.isKeyDown(SAPP_KEYCODE_LEFT_CONTROL)) *camera.mutablePosition() -= kWorldUp * kMoveSpeed * dt;

            const Eigen::Vector2f mouse_delta = input.mouseDelta();
            if (context.isCursorCaptured() && (mouse_delta.x() != 0.0f || mouse_delta.y() != 0.0f)) {
                yaw += mouse_delta.x() * kLookSensitivity * dt;
                pitch = std::clamp(pitch - mouse_delta.y() * kLookSensitivity * dt, -kPitchLimit, kPitchLimit);
            }
            *camera.mutableDirection() = Eigen::Vector3f{
                std::cos(pitch) * std::sin(yaw),
                std::sin(pitch),
                -std::cos(pitch) * std::cos(yaw),
            };
        }
};

} // namespace

int main(int argc, char** argv) {
    initLog(1024 * 1024);
    RLOG(LL_INFO, "Starting ElectronCloud");

    app::Application<ElectronCloud> app{{.width = 1920, .height = 1080, .title = "ElectronCloud", .resizable = false}, argc, argv};
    app.run();

    return 0;
}
