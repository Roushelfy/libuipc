#pragma once

#include <uipc/common/span.h>
#include <uipc/common/type_define.h>

namespace uipc::backend::cuda_mixed
{
struct StructuredChainShape
{
    SizeT horizon    = 0;
    SizeT block_size = 0;
    SizeT nrhs       = 1;
    bool  symmetric_positive_definite = false;
};

struct StructuredDofSlot
{
    IndexT old_dof   = -1;
    IndexT chain_dof = -1;
    SizeT  block     = 0;
    SizeT  lane      = 0;
    bool   is_padding = false;
    bool   scatter_write = true;
};

struct StructuredContributionStats
{
    SizeT  near_band_pair_count        = 0;
    SizeT  off_band_pair_count         = 0;
    SizeT  near_band_block_terms       = 0;
    SizeT  off_band_block_terms        = 0;
    double near_band_weighted_norm     = 0.0;
    double off_band_weighted_drop_norm = 0.0;
};

struct StructuredQualityReport
{
    double block_utilization = 0.0;
    double near_band_ratio   = 0.0;
    double off_band_ratio    = 0.0;
    SizeT  max_block_distance = 0;
    StructuredContributionStats contact_stats;
};

class StructuredAssemblySink
{
  public:
    virtual ~StructuredAssemblySink() = default;
};

class StructuredChainProvider
{
  public:
    virtual ~StructuredChainProvider() = default;

    virtual bool is_available() const = 0;
    virtual StructuredChainShape shape() const = 0;
    virtual span<const StructuredDofSlot> dof_slots() const = 0;
    virtual StructuredQualityReport quality_report() const = 0;
    virtual void assemble_chain(StructuredAssemblySink& sink) = 0;
};
}  // namespace uipc::backend::cuda_mixed
