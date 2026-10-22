#define RLOG_IMPLEMENTATION
#include <rlog.h>

#include <animation/timeline.hpp>
#include <app/application.hpp>
#include <app/cli_parser.hpp>
#include <graphics/presenter.hpp>
#include <graphics/recorder.hpp>
#include <graphics/primitives/scene/camera.hpp>
#include <renderer/electron_cloud_renderer.hpp>
#include <renderer/hydrogen.hpp>
#include <renderer/orbital.hpp>
#include <renderer/orbital_color.hpp>
#include <renderer/transition.hpp>

#include <common.h>

#include <Eigen/Dense>

#include <imgui.h>

#include <sokol_app.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>

// ElectronCloud: our own CUDA ray marcher (src/renderer/) plugged into GameEngine, in one of
// two modes picked on the command line.
//
// Live (default): shown through a graphics::Presenter, with a fly camera marking where the
// rays start, an FPS counter in the top-left corner and a debug panel to pick the orbital and
// play transitions between orbitals.
//   WASD move, Space / Left Ctrl up / down, ESC toggles mouse-look (release it to use the
//   debug panel).
//
// Record (--record <file.mp4>): a scripted animation (recordingScript()) rendered through a
// graphics::Recorder into a video, one frame every 1 / fps of animation time however long it
// takes to render, then the program exits. The window shows a preview.
//   ./run --record out.mp4 [--from 1s] [--to 2p_z] [--elevation 20] [--fps 60] [--width 1920] [--height 1080]
//   --elevation is the camera's angle above the x-z plane in degrees; 89 is a top view.
//   --intensity is the laser intensity in W/cm^2 (default 1e12): weaker light transitions more
//   slowly, so the cloud sloshes more times on the way (sloshes ~ 1 / sqrt(intensity)).

namespace {

using graphics::primitives::scene::Camera;

constexpr f32 kMoveSpeed = 2.0f;        // units/sec -- the scene is a few units across
constexpr f32 kLookSensitivity = 0.15f; // radians per pixel of mouse delta, per second
constexpr f32 kPitchLimit = 1.5533f;    // ~89 degrees
constexpr f32 kFovY = 1.0472f;          // 60 degrees

const Eigen::Vector3f kWorldUp{0.0f, 1.0f, 0.0f};

// The recording script's camera: an orbit around the nucleus at a fixed elevation, turning at a
// fixed rate, with its distance keyframed.
constexpr f32 kRecordSpinRate = 0.8f;          // radians per second of animation
constexpr f32 kDefaultElevationDegrees = 20.0f; // above the x-z plane
// The camera builds its sideways direction from world up, which is undefined looking straight
// along it, so a "top view" stops just short of 90 degrees (indistinguishable on screen).
constexpr f32 kMaxElevationDegrees = 89.0f;

// How far the recording camera orbits from the nucleus to frame a cloud of shell n: the
// cloud's reach (~1.8 / 6.6 / 13 A) fits inside the 60-degree view with some margin.
f32 cameraDistanceFor(i32 n) {
    switch (n) {
        case 1: return 6.0f;
        case 2: return 16.0f;
        default: return 26.0f;
    }
}

f32 aspectOf(const app::Context& context) {
    return context.height() > 0 ? static_cast<f32>(context.width()) / static_cast<f32>(context.height()) : 1.0f;
}

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

// A small see-through panel pinned to the top-left corner. ImGui's Framerate is averaged over
// the last 60 frames, so the number is steady enough to read.
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
            .states = {renderer::QuantumState{.n = 1, .l = 0, .m = 0}},
            .state_count = 1,
        };

        // A transition playing from one orbital to another (see transition.hpp). The physics
        // fixes how long it takes in atomic time (pi / Omega) and how many times the cloud
        // sloshes on the way; the animation only sets how fast that is played back.
        struct Transition {
                renderer::Orbital from;
                renderer::Orbital to;
                renderer::TransitionPlan plan;
                f64 playback_seconds; // real seconds the whole transition takes on screen
                f64 elapsed_seconds = 0.0;
        };
        std::optional<Transition> transition;

        // The transition controls in the debug panel.
        renderer::Orbital transition_target = renderer::Orbital::P2z;
        f32 transition_seconds = 5.0f;
        f32 laser_intensity = 1e12f; // W/cm^2; stronger light = faster Rabi frequency = fewer sloshes

        // Simulation time in atomic units: how far into the current transition we are, 0 when
        // none is playing.
        f64 simulation_time = 0.0;

        // How the cloud looks; brightness and absorption are tunable from the debug panel, and
        // cloud_color follows the orbital (orbital_color.hpp) every frame.
        renderer::Appearance appearance{
            .cloud_color = renderer::orbitalColor(renderer::Orbital::S1),
            .background_color = {0.02f, 0.02f, 0.03f},
            .absorption = 0.5f,
            .brightness = 1.5f,
        };

        renderer::ElectronCloudRenderer renderer{
            renderer::RaymarchSettings{.step_size = 0.02f, .max_distance = 100.0f, .min_density = 0.001f, .opacity_cutoff = 0.95f},
            appearance,
        };

        // Live mode shows frames through a Presenter; record mode through a Recorder, which also
        // writes them to the video. Exactly one is set.
        std::optional<graphics::Presenter> presenter;
        std::optional<graphics::Recorder> recorder;

        // Record mode's script: cues on a timeline plus the camera's keyframed distance, both
        // driven by frame index / fps.
        std::optional<animation::Timeline> timeline;
        std::optional<animation::Track<f32>> camera_distance;
        f32 camera_elevation = kDefaultElevationDegrees * std::numbers::pi_v<f32> / 180.0f; // radians
        u64 recorded_frames = 0;

        std::chrono::steady_clock::time_point last_time = std::chrono::steady_clock::now();

        explicit ElectronCloud(app::Context& context)
            : camera(kFovY, aspectOf(context), 0.1f, 100.0f, Eigen::Vector3f{0.0f, 0.0f, 3.0f}, Eigen::Vector3f{0.0f, 0.0f, -1.0f}, kWorldUp) {
            // DebugUI starts hidden; the FPS counter is always on.
            context.debugUI().setVisible(true);
            // Build the dipole table now (~0.2 s) rather than on the first frame that needs it.
            renderer::DipoleTable::get();

            app::CliParser cli{context.argc(), context.argv()};
            cli.defineArg<std::string>("record");
            cli.defineArg<std::string>("from");
            cli.defineArg<std::string>("to");
            cli.defineArg<f32>("elevation");
            cli.defineArg<f64>("intensity");
            cli.defineArg<u32>("fps");
            cli.defineArg<u32>("width");
            cli.defineArg<u32>("height");
            cli.parse();

            if (const std::optional<std::string> path = cli.arg<std::string>("record")) {
                graphics::Recorder::Config config{.path = *path};
                if (auto v = cli.arg<u32>("fps"); v && *v > 0) config.fps = *v;
                if (auto v = cli.arg<u32>("width"); v && *v > 0) config.width = *v;
                if (auto v = cli.arg<u32>("height"); v && *v > 0) config.height = *v;
                // The orbitals the script transitions between, by name ("1s", "2p_z", "3d_z2", ...).
                auto orbitalArg = [&](const char* name, renderer::Orbital fallback) {
                    const std::optional<std::string> text = cli.arg<std::string>(name);
                    if (!text) {
                        return fallback;
                    }
                    const std::optional<renderer::Orbital> orbital = renderer::orbitalFromName(*text);
                    if (!orbital || !renderer::hasWaveFunction(*orbital)) {
                        RLOG(LL_FATAL, "--%s %s: not an orbital with a wave function (1s, 2s, 2p_x, ..., 3d_x2-y2)", name, text->c_str());
                    }
                    return *orbital;
                };
                const renderer::Orbital from = orbitalArg("from", renderer::Orbital::S1);
                const renderer::Orbital to = orbitalArg("to", renderer::Orbital::P2z);
                if (from == to) {
                    RLOG(LL_FATAL, "--from and --to are both %s", renderer::orbitalName(from));
                }
                if (const std::optional<f32> degrees = cli.arg<f32>("elevation")) {
                    const f32 clamped = std::clamp(*degrees, -kMaxElevationDegrees, kMaxElevationDegrees);
                    if (clamped != *degrees) {
                        RLOG(LL_WARN, "--elevation %.1f: clamped to %.1f (the camera can't look straight along world up)", *degrees, clamped);
                    }
                    camera_elevation = clamped * std::numbers::pi_v<f32> / 180.0f;
                }
                if (const std::optional<f64> intensity = cli.arg<f64>("intensity")) {
                    if (*intensity <= 0.0) {
                        RLOG(LL_FATAL, "--intensity must be > 0 W/cm^2");
                    }
                    laser_intensity = static_cast<f32>(*intensity);
                }
                recorder.emplace(config);
                recordingScript(from, to);
            } else {
                presenter.emplace();
            }
        }

        void frame(app::Context& context) {
            if (recorder) {
                recordFrame(context);
            } else {
                liveFrame(context);
            }
        }

        // The recording: spin around `from` for 4 s, keep spinning through a 10 s transition to
        // `to` (4 s -> 14 s), spin around `to` for 7 more seconds, stop at 21 s. The camera's
        // distance eases between the two clouds' sizes during the transition.
        void recordingScript(renderer::Orbital from, renderer::Orbital to) {
            constexpr f64 kInitialSpin = 4.0;
            constexpr f64 kTransition = 10.0;
            constexpr f64 kFinalSpin = 7.0;
            constexpr f64 kTransitionStart = kInitialSpin;
            constexpr f64 kTransitionEnd = kTransitionStart + kTransition;
            constexpr f64 kEnd = kTransitionEnd + kFinalSpin;

            selectOrbital(from);
            transition_seconds = static_cast<f32>(kTransition);
            const f32 near = cameraDistanceFor(renderer::quantumNumbersOf(from).n);
            const f32 far = cameraDistanceFor(renderer::quantumNumbersOf(to).n);
            camera_distance.emplace(std::vector<animation::Key<f32>>{
                {0.0, near}, {kTransitionStart, near}, {kTransitionEnd, far}, {kEnd, far}});
            timeline.emplace(kEnd, std::vector<animation::Timeline::Cue>{
                                       {kTransitionStart, "transition", [this, to]() { startTransition(to); }},
                                   });
            RLOG(LL_INFO, "recording %s -> %s%s", renderer::orbitalName(from), renderer::orbitalName(to),
                 renderer::DipoleTable::get().allowed(from, to) ? "" : " (forbidden: animated with a stand-in dipole)");
            const renderer::TransitionPlan plan = renderer::planTransition(from, to, atom.nuclear_charge, laser_intensity);
            RLOG(LL_INFO, "laser %.2e W/cm^2: the transition takes %.3g fs and sloshes %.1f times", static_cast<f64>(laser_intensity),
                 plan.duration * renderer::kAtomicUnitTimeSeconds * 1e15, plan.sloshes);
        }

        // One frame of the recording, at exactly recorded_frames / fps into the script.
        void recordFrame(app::Context& context) {
            const f64 dt = recorder->frameTime();
            const f64 time = static_cast<f64>(recorded_frames) * dt;

            // A playing transition moves first, so one a cue starts this frame shows its first
            // moment (pure start orbital) rather than one step in.
            advanceTransition(dt);
            timeline->setTime(time);
            updateCloudColor();

            // Orbit the nucleus, always looking at it.
            const f32 azimuth = kRecordSpinRate * static_cast<f32>(time);
            const f32 distance = camera_distance->sample(time);
            const Eigen::Vector3f offset{std::cos(camera_elevation) * std::sin(azimuth), std::sin(camera_elevation), std::cos(camera_elevation) * std::cos(azimuth)};
            *camera.mutablePosition() = atom.nucleus + offset * distance;
            *camera.mutableDirection() = -offset;
            camera.setAspectRatio(static_cast<f32>(recorder->config().width) / static_cast<f32>(recorder->config().height));

            renderer.beginFrame(camera, simulation_time);
            renderer.submit(atom);
            recorder->present(renderer, [&]() {
                context.debugUI().draw(context.width(), context.height(), dt, sapp_dpi_scale(), [&]() {
                    drawFpsCounter();
                    drawRecordingPanel(time);
                });
            });
            recorded_frames++;

            const u64 total_frames = static_cast<u64>(std::llround(timeline->duration() / dt));
            if (recorded_frames >= total_frames) {
                recorder->finish();
                context.quit();
            }
        }

        // Below the FPS counter while recording: which frame of how many, and where in the script.
        void drawRecordingPanel(f64 time) {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 50.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Recording", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                const u64 total_frames = static_cast<u64>(std::llround(timeline->duration() / recorder->frameTime()));
                char label[64];
                std::snprintf(label, sizeof(label), "frame %llu / %llu", static_cast<unsigned long long>(recorded_frames + 1),
                              static_cast<unsigned long long>(total_frames));
                ImGui::ProgressBar(static_cast<f32>(recorded_frames + 1) / static_cast<f32>(total_frames), ImVec2(260.0f, 0.0f), label);
                ImGui::Text("t = %.2f s of %.2f s, %u fps", time, timeline->duration(), recorder->config().fps);
                ImGui::Text("%s", recorder->config().path.c_str());
                if (transition) {
                    ImGui::Text("%s -> %s", renderer::orbitalName(transition->from), renderer::orbitalName(transition->to));
                } else {
                    ImGui::Text("%s", renderer::orbitalName(currentOrbital()));
                }
            }
            ImGui::End();
        }

        // One frame of live mode: wall-clock time, fly camera, debug panel.
        void liveFrame(app::Context& context) {
            const auto now = std::chrono::steady_clock::now();
            const f32 dt = std::chrono::duration<f32>(now - last_time).count();
            last_time = now;

            if (context.input().isKeyPressed(SAPP_KEYCODE_ESCAPE)) {
                context.toggleCursorCapture();
            }
            camera.setAspectRatio(aspectOf(context));
            flyCamera(context, dt);
            advanceTransition(dt);
            updateCloudColor();

            renderer.beginFrame(camera, simulation_time);
            renderer.submit(atom);
            presenter->present(renderer, [&]() {
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
            atom.states[0] = renderer::QuantumState{.n = numbers.n, .l = numbers.l, .m = numbers.m};
            atom.state_count = 1;
            atom.superposition = renderer::Superposition{};
            atom.bounds_radius = boundsRadiusFor(numbers.n);
        }

        renderer::Orbital currentOrbital() const {
            const auto& state = atom.states[0];
            return renderer::orbitalFor(state.n, state.l, state.m).value_or(renderer::Orbital::Count);
        }

        // Starts the current orbital -> `target`, driven by light of laser_intensity, played back
        // over transition_seconds. The atom becomes a two-state superposition until it ends.
        void startTransition(renderer::Orbital target) {
            const renderer::Orbital from = currentOrbital();
            const renderer::TransitionPlan plan = renderer::planTransition(from, target, atom.nuclear_charge, laser_intensity);
            if (target == from || plan.duration <= 0.0) {
                return;
            }
            const renderer::QuantumNumbers numbers = renderer::quantumNumbersOf(target);
            atom.states[1] = renderer::QuantumState{.n = numbers.n, .l = numbers.l, .m = numbers.m};
            atom.state_count = 2;
            atom.superposition = renderer::Superposition{};
            atom.bounds_radius = std::max(atom.bounds_radius, boundsRadiusFor(numbers.n));
            transition = Transition{.from = from, .to = target, .plan = plan, .playback_seconds = transition_seconds};
            simulation_time = 0.0;
        }

        // Moves a playing transition on by `dt` seconds of playback and updates the atom's
        // weights. When it reaches the end the atom is left in the target orbital alone, which is
        // exactly what the superposition looks like at that moment, so nothing jumps.
        void advanceTransition(f64 dt) {
            if (!transition) {
                simulation_time = 0.0;
                return;
            }
            transition->elapsed_seconds += dt;
            const f64 progress = std::min(transition->elapsed_seconds / transition->playback_seconds, 1.0);
            simulation_time = progress * transition->plan.duration;
            atom.superposition = renderer::superpositionAt(transition->plan, simulation_time);
            // A tiny tolerance: in record mode the elapsed time is a sum of 1 / fps steps, which
            // can land a hair under the duration on the frame that should end it.
            if (progress >= 1.0 - 1e-9) {
                const renderer::Orbital to = transition->to;
                transition.reset();
                selectOrbital(to);
                simulation_time = 0.0;
            }
        }

        // The current orbital's colour; during a transition, a fade from the start orbital's
        // hue to the target's by the target's population -- the same fraction the density has
        // moved -- so colour and shape arrive together.
        void updateCloudColor() {
            const Eigen::Vector3f color = transition ? renderer::blendOrbitalColors(transition->from, transition->to, atom.superposition.b)
                                                     : renderer::orbitalColor(currentOrbital());
            if (color != appearance.cloud_color) {
                appearance.cloud_color = color;
                renderer.setAppearance(appearance);
            }
        }

        // An orbital's colour as an ImGui text colour, so the menus show which is which.
        static ImVec4 orbitalTextColor(renderer::Orbital orbital) {
            const Eigen::Vector3f color = renderer::orbitalColor(orbital);
            return ImVec4(color.x(), color.y(), color.z(), 1.0f);
        }

        void cancelTransition() {
            if (transition) {
                const renderer::Orbital from = transition->from;
                transition.reset();
                selectOrbital(from);
                simulation_time = 0.0;
            }
        }

        // "2p_z", "2s  (forbidden)" or "2p_z  (dE = 0)" for a target of the current orbital.
        static void transitionLabel(char* out, usize size, renderer::Orbital from, renderer::Orbital to) {
            const char* flag = "";
            if (!renderer::DipoleTable::get().allowed(from, to)) {
                flag = "  (forbidden)";
            } else if (renderer::quantumNumbersOf(from).n == renderer::quantumNumbersOf(to).n) {
                flag = "  (dE = 0)";
            }
            std::snprintf(out, size, "%s%s", renderer::orbitalName(to), flag);
        }

        // Below the FPS counter: an orbital dropdown, one at a time, listing only the orbitals
        // that have a wave function.
        void drawDebugPanel() {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 50.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                const renderer::Orbital current = currentOrbital();
                ImGui::BeginDisabled(transition.has_value());
                if (ImGui::BeginCombo("Orbital", renderer::orbitalName(current))) {
                    for (u32 i = 0; i < static_cast<u32>(renderer::Orbital::Count); i++) {
                        const auto orbital = static_cast<renderer::Orbital>(i);
                        if (!renderer::hasWaveFunction(orbital)) {
                            continue;
                        }
                        const bool selected = orbital == current;
                        ImGui::PushStyleColor(ImGuiCol_Text, orbitalTextColor(orbital));
                        const bool clicked = ImGui::Selectable(renderer::orbitalName(orbital), selected);
                        ImGui::PopStyleColor();
                        if (clicked) {
                            selectOrbital(orbital);
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::EndDisabled();
                const renderer::QuantumNumbers numbers = renderer::quantumNumbersOf(current);
                ImGui::Text("n = %d, l = %d, m = %d", numbers.n, numbers.l, numbers.m);

                ImGui::Separator();
                drawTransitionControls(current);

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

        // Pick a target, a playback duration and the laser intensity, then start; while one
        // plays, a progress bar and a cancel button instead. Every orbital is a target --
        // forbidden ones are flagged and animated as if they had a typical dipole.
        void drawTransitionControls(renderer::Orbital current) {
            ImGui::TextUnformatted("Transition");

            if (transition) {
                const f64 progress = std::min(transition->elapsed_seconds / transition->playback_seconds, 1.0);
                char label[64];
                std::snprintf(label, sizeof(label), "%s -> %s", renderer::orbitalName(transition->from), renderer::orbitalName(transition->to));
                ImGui::ProgressBar(static_cast<f32>(progress), ImVec2(-1.0f, 0.0f), label);
                ImGui::Text("populations: %.2f / %.2f", atom.superposition.a, atom.superposition.b);
                ImGui::Text("t = %.1f a.u. (%.3g fs) of %.1f a.u.", simulation_time, simulation_time * renderer::kAtomicUnitTimeSeconds * 1e15,
                            transition->plan.duration);
                if (ImGui::Button("Cancel")) {
                    cancelTransition();
                }
                return;
            }

            if (transition_target == current) {
                transition_target = current == renderer::Orbital::S1 ? renderer::Orbital::P2z : renderer::Orbital::S1;
            }
            char preview[64];
            transitionLabel(preview, sizeof(preview), current, transition_target);
            if (ImGui::BeginCombo("To", preview)) {
                for (u32 i = 0; i < static_cast<u32>(renderer::Orbital::Count); i++) {
                    const auto orbital = static_cast<renderer::Orbital>(i);
                    if (!renderer::hasWaveFunction(orbital) || orbital == current) {
                        continue;
                    }
                    char item[64];
                    transitionLabel(item, sizeof(item), current, orbital);
                    const bool selected = orbital == transition_target;
                    ImGui::PushStyleColor(ImGuiCol_Text, orbitalTextColor(orbital));
                    const bool clicked = ImGui::Selectable(item, selected);
                    ImGui::PopStyleColor();
                    if (clicked) {
                        transition_target = orbital;
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SliderFloat("Duration (s)", &transition_seconds, 0.5f, 60.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Laser (W/cm^2)", &laser_intensity, 1e6f, 1e14f, "%.1e", ImGuiSliderFlags_Logarithmic);

            // What the physics says about this transition at this intensity.
            const renderer::TransitionPlan plan = renderer::planTransition(current, transition_target, atom.nuclear_charge, laser_intensity);
            if (plan.allowed) {
                ImGui::Text("dipole %.3f e a0", plan.dipole);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "forbidden: animated as if dipole = %.1f e a0", plan.dipole);
            }
            ImGui::Text("takes %.3g fs, sloshes %.1f times", plan.duration * renderer::kAtomicUnitTimeSeconds * 1e15, plan.sloshes);
            // Each slosh needs a few frames to be seen; past that the sloshing aliases into flicker.
            const f32 fps = ImGui::GetIO().Framerate;
            const f64 frames_per_slosh = plan.sloshes > 0.0 ? transition_seconds * fps / plan.sloshes : 0.0;
            if (plan.sloshes > 0.0 && frames_per_slosh < 4.0) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "%.1f frames per slosh: too fast to see --", frames_per_slosh);
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "raise the duration or the intensity");
            }

            if (ImGui::Button("Start transition")) {
                startTransition(transition_target);
            }
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
