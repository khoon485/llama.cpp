// common/progress.h - Progress bar utilities (tqdm style)
#pragma once

#include <string>
#include <iostream>
#include <iomanip>

namespace training {

inline std::string format_time(double seconds) {
    int s = (int)seconds;
    if (s < 60) return std::to_string(s) + "s";
    if (s < 3600) return std::to_string(s / 60) + ":" + (s % 60 < 10 ? "0" : "") + std::to_string(s % 60);
    int h = s / 3600;
    int m = (s % 3600) / 60;
    return std::to_string(h) + ":" + (m < 10 ? "0" : "") + std::to_string(m) + ":" + (s % 60 < 10 ? "0" : "") + std::to_string(s % 60);
}

// DPO style progress (with loss, margin, accuracy)
inline void print_progress_dpo(
    int epoch, int n_epochs, int current, int total,
    float loss, float margin, double elapsed_sec,
    float accuracy = -1.0f, const char * phase = nullptr
) {
    float percent = (float)current / total * 100.0f;
    int bar_width = 20;
    int pos = (int)(bar_width * percent / 100.0f);

    double per_sample = (current > 0) ? elapsed_sec / current : 0;
    double eta_sec = per_sample * (total - current);
    double it_per_sec = (elapsed_sec > 0) ? current / elapsed_sec : 0;

    std::cout << "\r";
    if (phase) {
        std::cout << phase << " [";
    } else {
        std::cout << "E" << (epoch + 1) << "/" << n_epochs << " [";
    }

    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << std::setw(3) << (int)percent << "% ";
    std::cout << current << "/" << total;
    std::cout << " [" << format_time(elapsed_sec) << "<" << format_time(eta_sec);
    std::cout << ", " << std::fixed << std::setprecision(2) << it_per_sec << "it/s]";

    if (loss > 0) {
        std::cout << " L:" << std::setprecision(3) << loss;
        std::cout << " M:" << std::setprecision(2) << margin;
        if (accuracy >= 0) {
            std::cout << " A:" << std::setprecision(0) << (accuracy * 100.0f) << "%";
        }
    }

    std::cout << std::flush;
}

// CE style progress (with loss only)
inline void print_progress_ce(
    int epoch, int n_epochs, int current, int total,
    float loss, double elapsed_sec, const char * phase = nullptr
) {
    float percent = (float)current / total * 100.0f;
    int bar_width = 20;
    int pos = (int)(bar_width * percent / 100.0f);

    double per_sample = (current > 0) ? elapsed_sec / current : 0;
    double eta_sec = per_sample * (total - current);
    double it_per_sec = (elapsed_sec > 0) ? current / elapsed_sec : 0;

    std::cout << "\r";
    if (phase) {
        std::cout << phase << " [";
    } else {
        std::cout << "E" << (epoch + 1) << "/" << n_epochs << " [";
    }

    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << std::setw(3) << (int)percent << "% ";
    std::cout << current << "/" << total;
    std::cout << " [" << format_time(elapsed_sec) << "<" << format_time(eta_sec);
    std::cout << ", " << std::fixed << std::setprecision(2) << it_per_sec << "it/s]";

    if (loss > 0) {
        std::cout << " L:" << std::setprecision(4) << loss;
    }

    std::cout << std::flush;
}

} // namespace training
