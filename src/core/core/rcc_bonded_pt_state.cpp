#include <uipc/core/rcc_bonded_pt_state.h>

#include <algorithm>
#include <numeric>

namespace uipc::core
{
void RCCBondedPTState::clear()
{
    m_locked_keys.clear();
    m_locked_topos.clear();
    m_locked_beta.clear();
    m_locked_age.clear();
    m_release_flags.clear();
    sync_locked_count();
}

void RCCBondedPTState::reserve(SizeT size)
{
    m_locked_keys.reserve(size);
    m_locked_topos.reserve(size);
    m_locked_beta.reserve(size);
    m_locked_age.reserve(size);
    m_release_flags.reserve(size);
}

void RCCBondedPTState::push_locked(const RCCBondedPTEntry& entry)
{
    m_locked_keys.push_back(entry.key);
    m_locked_topos.push_back(entry.topo);
    m_locked_beta.push_back(entry.beta);
    m_locked_age.push_back(entry.age);
    m_release_flags.push_back(entry.release_flags);
    sync_locked_count();
}

SizeT RCCBondedPTState::size() const
{
    return m_locked_keys.size();
}

bool RCCBondedPTState::empty() const
{
    return m_locked_keys.empty();
}

bool RCCBondedPTState::validate() const
{
    const SizeT n = m_locked_keys.size();
    return m_locked_topos.size() == n && m_locked_beta.size() == n
           && m_locked_age.size() == n && m_release_flags.size() == n;
}

const RCCBondedPTCounters& RCCBondedPTState::counters() const
{
    return m_counters;
}

span<const U64> RCCBondedPTState::locked_keys() const
{
    return m_locked_keys;
}

span<const Vector4i> RCCBondedPTState::locked_topos() const
{
    return m_locked_topos;
}

span<const Float> RCCBondedPTState::locked_beta() const
{
    return m_locked_beta;
}

span<const IndexT> RCCBondedPTState::locked_age() const
{
    return m_locked_age;
}

span<const U32> RCCBondedPTState::release_flags() const
{
    return m_release_flags;
}

RCCBondedPTEntry RCCBondedPTState::entry(SizeT index) const
{
    return RCCBondedPTEntry{m_locked_keys[index],
                            m_locked_topos[index],
                            m_locked_beta[index],
                            m_locked_age[index],
                            m_release_flags[index]};
}

SizeT RCCBondedPTState::find_key(U64 key) const
{
    const auto it = std::lower_bound(m_locked_keys.begin(), m_locked_keys.end(), key);
    if(it == m_locked_keys.end() || *it != key)
        return npos;
    return static_cast<SizeT>(std::distance(m_locked_keys.begin(), it));
}

void RCCBondedPTState::sort_by_key()
{
    vector<SizeT> order(size());
    std::iota(order.begin(), order.end(), SizeT{0});
    std::stable_sort(order.begin(), order.end(), [&](SizeT lhs, SizeT rhs)
    {
        return m_locked_keys[lhs] < m_locked_keys[rhs];
    });

    vector<U64>      keys;
    vector<Vector4i> topos;
    vector<Float>    beta;
    vector<IndexT>   age;
    vector<U32>      flags;
    keys.reserve(order.size());
    topos.reserve(order.size());
    beta.reserve(order.size());
    age.reserve(order.size());
    flags.reserve(order.size());

    for(SizeT i : order)
    {
        keys.push_back(m_locked_keys[i]);
        topos.push_back(m_locked_topos[i]);
        beta.push_back(m_locked_beta[i]);
        age.push_back(m_locked_age[i]);
        flags.push_back(m_release_flags[i]);
    }

    m_locked_keys   = std::move(keys);
    m_locked_topos  = std::move(topos);
    m_locked_beta   = std::move(beta);
    m_locked_age    = std::move(age);
    m_release_flags = std::move(flags);
    sync_locked_count();
}

bool RCCBondedPTState::mark_released(U64 key, U32 release_flags)
{
    const SizeT index = find_key(key);
    if(index == npos)
        return false;
    m_release_flags[index] |= release_flags;
    return true;
}

vector<RCCBondedPTEntry> RCCBondedPTState::extract_released()
{
    vector<RCCBondedPTEntry> released;
    vector<U64>              keep_keys;
    vector<Vector4i>         keep_topos;
    vector<Float>            keep_beta;
    vector<IndexT>           keep_age;
    vector<U32>              keep_flags;

    released.reserve(size());
    keep_keys.reserve(size());
    keep_topos.reserve(size());
    keep_beta.reserve(size());
    keep_age.reserve(size());
    keep_flags.reserve(size());

    for(SizeT i = 0; i < size(); ++i)
    {
        const RCCBondedPTEntry e = entry(i);
        if(e.release_flags != RCCBondedPTReleaseNone)
        {
            released.push_back(e);
        }
        else
        {
            keep_keys.push_back(e.key);
            keep_topos.push_back(e.topo);
            keep_beta.push_back(e.beta);
            keep_age.push_back(e.age);
            keep_flags.push_back(e.release_flags);
        }
    }

    m_locked_keys   = std::move(keep_keys);
    m_locked_topos  = std::move(keep_topos);
    m_locked_beta   = std::move(keep_beta);
    m_locked_age    = std::move(keep_age);
    m_release_flags = std::move(keep_flags);
    m_counters.released_count += released.size();
    sync_locked_count();
    return released;
}

void RCCBondedPTState::clear_counters()
{
    m_counters = {};
    sync_locked_count();
}

void RCCBondedPTState::set_counters(const RCCBondedPTCounters& counters)
{
    m_counters = counters;
    sync_locked_count();
}

void RCCBondedPTState::record_candidates(SizeT count)
{
    m_counters.candidate_count += count;
}

void RCCBondedPTState::record_degenerate_rejected(SizeT count)
{
    m_counters.degenerate_rejected_count += count;
}

void RCCBondedPTState::record_filter_skipped(SizeT count)
{
    m_counters.filter_skipped_count += count;
}

void RCCBondedPTState::record_duplicate_suppressed(SizeT count)
{
    m_counters.duplicate_suppressed_count += count;
}

void RCCBondedPTState::sync_locked_count()
{
    m_counters.locked_count = size();
}
}  // namespace uipc::core
