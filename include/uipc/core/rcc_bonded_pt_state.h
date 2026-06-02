#pragma once
#include <uipc/common/dllexport.h>
#include <uipc/common/span.h>
#include <uipc/common/type_define.h>
#include <uipc/common/vector.h>

#include <limits>

namespace uipc::core
{
enum RCCBondedPTReleaseFlag : U32
{
    RCCBondedPTReleaseNone        = 0,
    RCCBondedPTReleaseStrain      = 1u << 0,
    RCCBondedPTReleaseGap         = 1u << 1,
    RCCBondedPTReleaseSlip        = 1u << 2,
    RCCBondedPTReleaseFlip        = 1u << 3,
    RCCBondedPTReleaseStickySide  = 1u << 4,
    RCCBondedPTReleasePolicy      = 1u << 5,
    RCCBondedPTReleaseDegenerate  = 1u << 6,
};

struct UIPC_CORE_API RCCBondedPTEntry
{
    U64      key = 0;
    Vector4i topo = Vector4i::Zero();
    Float    beta = 0.0;
    IndexT   age  = 0;
    U32      release_flags = RCCBondedPTReleaseNone;
    Matrix3x3 Dm_inv = Matrix3x3::Identity();
    Float     rest_volume = 0.0;
};

struct UIPC_CORE_API RCCBondedPTCounters
{
    SizeT candidate_count = 0;
    SizeT locked_count = 0;
    SizeT released_count = 0;
    SizeT degenerate_rejected_count = 0;
    SizeT filter_skipped_count = 0;
    SizeT duplicate_suppressed_count = 0;
};

class UIPC_CORE_API RCCBondedPTState
{
  public:
    static constexpr SizeT npos = std::numeric_limits<SizeT>::max();

    void clear();
    void reserve(SizeT size);
    void push_locked(const RCCBondedPTEntry& entry);

    SizeT size() const;
    bool  empty() const;
    bool  validate() const;
    const RCCBondedPTCounters& counters() const;

    span<const U64>      locked_keys() const;
    span<const Vector4i> locked_topos() const;
    span<const Float>    locked_beta() const;
    span<const IndexT>   locked_age() const;
    span<const U32>      release_flags() const;
    span<const Matrix3x3> locked_dm_inv() const;
    span<const Float>     locked_rest_volume() const;

    RCCBondedPTEntry entry(SizeT index) const;

    SizeT find_key(U64 key) const;
    void  sort_by_key();
    bool  mark_released(U64 key, U32 release_flags);
    vector<RCCBondedPTEntry> extract_released();

    void clear_counters();
    void set_counters(const RCCBondedPTCounters& counters);
    void record_candidates(SizeT count);
    void record_degenerate_rejected(SizeT count);
    void record_filter_skipped(SizeT count);
    void record_duplicate_suppressed(SizeT count);

  private:
    void sync_locked_count();

    vector<U64>      m_locked_keys;
    vector<Vector4i> m_locked_topos;
    vector<Float>    m_locked_beta;
    vector<IndexT>   m_locked_age;
    vector<U32>      m_release_flags;
    vector<Matrix3x3> m_locked_dm_inv;
    vector<Float>     m_locked_rest_volume;
    RCCBondedPTCounters m_counters;
};
}  // namespace uipc::core
