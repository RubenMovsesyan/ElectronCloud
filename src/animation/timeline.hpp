#pragma once

// A minimal keyframe system for scripted recordings: Tracks of values eased between keys, and
// a Timeline of one-off Cues fired as time passes. Time is whatever the caller drives it with
// -- in record mode, frame index / fps, so a recording is deterministic and independent of
// how long frames take to render.

#include <common.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace animation {

enum class Ease {
    Linear,
    Smooth, // smoothstep: eases in and out, so a value starts and stops moving gently
};

template <typename T>
struct Key {
    f64 time; // seconds
    T value;
};

// A value over time: held at the first key's value before it, the last key's after it, and
// eased between neighbouring keys. T needs `T + T`, `T - T` and `T * f32` (f32, Eigen vectors).
template <typename T>
class Track {
    public:
        Track(std::vector<Key<T>> keys, Ease ease = Ease::Smooth) : m_keys(std::move(keys)), m_ease(ease) {
            std::ranges::sort(m_keys, {}, &Key<T>::time);
        }

        T sample(f64 time) const {
            if (m_keys.empty()) {
                return T{};
            }
            if (time <= m_keys.front().time) {
                return m_keys.front().value;
            }
            if (time >= m_keys.back().time) {
                return m_keys.back().value;
            }
            const auto next = std::ranges::upper_bound(m_keys, time, {}, &Key<T>::time);
            const Key<T>& b = *next;
            const Key<T>& a = *(next - 1);
            f32 t = static_cast<f32>((time - a.time) / (b.time - a.time));
            if (m_ease == Ease::Smooth) {
                t = t * t * (3.0f - 2.0f * t);
            }
            return a.value + (b.value - a.value) * t;
        }

    private:
        std::vector<Key<T>> m_keys;
        Ease m_ease;
};

// One-off actions at given times plus an end time. setTime() fires every cue at or before the
// new time that hasn't fired yet, in time order, so jumping ahead still runs them all.
class Timeline {
    public:
        struct Cue {
                f64 time;
                std::string label; // for logs
                std::function<void()> action;
        };

        Timeline(f64 duration, std::vector<Cue> cues) : m_duration(duration), m_cues(std::move(cues)) {
            std::ranges::sort(m_cues, {}, &Cue::time);
        }

        void setTime(f64 time) {
            m_time = time;
            while (m_next_cue < m_cues.size() && m_cues[m_next_cue].time <= m_time) {
                m_cues[m_next_cue].action();
                m_next_cue++;
            }
        }

        f64 time() const { return m_time; }
        f64 duration() const { return m_duration; }
        bool finished() const { return m_time >= m_duration; }

    private:
        f64 m_duration;
        std::vector<Cue> m_cues;
        usize m_next_cue = 0;
        f64 m_time = 0.0;
};

} // namespace animation
