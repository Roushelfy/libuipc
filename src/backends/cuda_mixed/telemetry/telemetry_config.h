#pragma once
#include <cstddef>
#include <string>

namespace uipc::backend::cuda_mixed
{
struct TelemetryConfig
{
    struct TimerOptions
    {
        bool enable = false;
        bool report_every_frame = false;
    };

    struct NvtxOptions
    {
        bool enable = false;
    };

    struct PcgOptions
    {
        bool   enable            = false;
        size_t sample_every_iter = 10;
    };

    struct ErrorTrackerOptions
    {
        bool        enable        = false;
        std::string mode          = "offline";
        std::string reference_dir = "";
    };

    bool                enable = false;
    TimerOptions        timer;
    NvtxOptions         nvtx;
    PcgOptions          pcg;
    ErrorTrackerOptions error_tracker;
};
}  // namespace uipc::backend::cuda_mixed

