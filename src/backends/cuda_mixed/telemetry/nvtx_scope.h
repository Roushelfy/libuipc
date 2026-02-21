#pragma once

#ifdef UIPC_WITH_NVTX
#include <muda/profiler.h>
#define UIPC_NVTX_RANGE(name) muda::RangeName _uipc_nvtx_range_##__LINE__ {name}
#else
#define UIPC_NVTX_RANGE(name) ((void)0)
#endif

