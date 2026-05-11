#include <app/app.h>
#include <linear_system/socu_contact_topology_stamp.h>

#include <cuda_runtime.h>
#include <muda/buffer/device_buffer.h>

#include <utility>
#include <vector>

namespace
{
using namespace uipc::backend::cuda_mixed;
using uipc::SizeT;
using uipc::Vector4i;

bool has_cuda_device()
{
    int device_count = 0;
    const cudaError_t device_query = cudaGetDeviceCount(&device_count);
    if(device_query != cudaSuccess || device_count == 0)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

SocuContactTopologyStamp finalize_stamp(
    SocuContactTopologyStamp           stamp,
    SocuContactTopologyHashWorkspace&  workspace,
    cudaStream_t                       stream,
    SizeT                              reporter_count)
{
    socu_contact_topology_finalize_metadata(stamp, reporter_count);
    if(stamp.source_count != 0)
    {
        const auto device_hash =
            socu_contact_topology_hash_finish(workspace, stream);
        socu_contact_topology_mix_device_hash(stamp, device_hash);
    }
    return stamp;
}

SocuContactTopologyStamp build_pt_stamp(const std::vector<Vector4i>& pts,
                                        SizeT layout_token)
{
    muda::DeviceBuffer<Vector4i> device_pts{pts};
    SocuContactTopologyHashWorkspace workspace;
    SocuContactTopologyStamp         stamp =
        socu_contact_topology_make_stamp_seed();
    stamp.counts.simplex_normal_pt = pts.size();

    socu_contact_topology_hash_reset(workspace, cudaStreamLegacy);
    socu_contact_topology_mix_view(stamp,
                                   workspace,
                                   cudaStreamLegacy,
                                   SizeT{0},
                                   SizeT{0},
                                   SocuContactSourceFamily::SimplexNormalPT,
                                   std::as_const(device_pts).view(),
                                   layout_token);
    return finalize_stamp(stamp, workspace, cudaStreamLegacy, SizeT{1});
}

SocuContactTopologyStamp build_two_source_stamp(bool swapped)
{
    std::vector<Vector4i> pt = {Vector4i{1, 2, 3, 4}};
    std::vector<Vector4i> ee = {Vector4i{5, 6, 7, 8}};
    muda::DeviceBuffer<Vector4i> device_pt{pt};
    muda::DeviceBuffer<Vector4i> device_ee{ee};

    SocuContactTopologyHashWorkspace workspace;
    SocuContactTopologyStamp         stamp =
        socu_contact_topology_make_stamp_seed();
    stamp.counts.simplex_normal_pt = 1;
    stamp.counts.simplex_normal_ee = 1;

    socu_contact_topology_hash_reset(workspace, cudaStreamLegacy);
    if(swapped)
    {
        socu_contact_topology_mix_view(stamp,
                                       workspace,
                                       cudaStreamLegacy,
                                       SizeT{0},
                                       SizeT{0},
                                       SocuContactSourceFamily::SimplexNormalEE,
                                       std::as_const(device_ee).view(),
                                       SizeT{202});
        socu_contact_topology_mix_view(stamp,
                                       workspace,
                                       cudaStreamLegacy,
                                       SizeT{0},
                                       SizeT{1},
                                       SocuContactSourceFamily::SimplexNormalPT,
                                       std::as_const(device_pt).view(),
                                       SizeT{101});
    }
    else
    {
        socu_contact_topology_mix_view(stamp,
                                       workspace,
                                       cudaStreamLegacy,
                                       SizeT{0},
                                       SizeT{0},
                                       SocuContactSourceFamily::SimplexNormalPT,
                                       std::as_const(device_pt).view(),
                                       SizeT{101});
        socu_contact_topology_mix_view(stamp,
                                       workspace,
                                       cudaStreamLegacy,
                                       SizeT{0},
                                       SizeT{1},
                                       SocuContactSourceFamily::SimplexNormalEE,
                                       std::as_const(device_ee).view(),
                                       SizeT{202});
    }
    return finalize_stamp(stamp, workspace, cudaStreamLegacy, SizeT{1});
}

SocuAssemblyPlanKey plan_key_from_stamp(const SocuContactTopologyStamp& stamp)
{
    SocuAssemblyPlanKey key;
    key.ordering_epoch = 7;
    key.native_descriptor_epoch = 11;
    key.contact_topology_epoch = stamp.epoch;
    key.contact_layout_hash = stamp.layout_hash;
    key.contact_content_hash = stamp.content_hash;
    key.fixed_mapping_epoch = 13;
    key.vertex_projection_epoch = 17;
    key.horizon = 5;
    key.block_size = 16;
    return key;
}
}  // namespace

TEST_CASE("cuda_mixed_socu_contact_topology_device_hash_producer",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact topology stamp tests");

    const std::vector<Vector4i> topology_a = {Vector4i{1, 2, 3, 4},
                                              Vector4i{5, 6, 7, 8}};
    const std::vector<Vector4i> topology_b = {Vector4i{1, 2, 3, 9},
                                              Vector4i{5, 6, 7, 8}};

    auto stamp_a = build_pt_stamp(topology_a, SizeT{77});
    auto stamp_b = build_pt_stamp(topology_b, SizeT{77});

    CHECK(stamp_a.layout_hash == stamp_b.layout_hash);
    CHECK(stamp_a.content_hash != stamp_b.content_hash);

    SocuContactTopologyStampCache cache;
    const auto cached_a = cache.update(stamp_a);
    const auto cached_b = cache.update(stamp_b);
    CHECK(cached_a.epoch == SizeT{1});
    CHECK(cached_b.epoch == SizeT{2});

    SocuContactTopologyStampCache stable_cache;
    const auto stable_a0 = stable_cache.update(stamp_a);
    const auto stable_a1 = stable_cache.update(stamp_a);
    CHECK(stable_a0.epoch == SizeT{1});
    CHECK(stable_a1.epoch == stable_a0.epoch);
}

TEST_CASE("cuda_mixed_socu_contact_topology_empty_family_stamp",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact topology stamp tests");

    const std::vector<Vector4i> empty;
    auto stamp = build_pt_stamp(empty, SizeT{88});
    CHECK(stamp.counts.simplex_normal_pt == SizeT{0});
    CHECK(stamp.source_count == SizeT{1});
    CHECK(stamp.reporter_count == SizeT{1});

    SocuContactTopologyStampCache cache;
    const auto first = cache.update(stamp);
    const auto second = cache.update(stamp);
    CHECK(first.valid());
    CHECK(first.epoch == second.epoch);
}

TEST_CASE("cuda_mixed_socu_contact_topology_order_and_family_sensitivity",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact topology stamp tests");

    const std::vector<Vector4i> ordered = {Vector4i{1, 2, 3, 4},
                                           Vector4i{5, 6, 7, 8}};
    const std::vector<Vector4i> reordered = {Vector4i{5, 6, 7, 8},
                                             Vector4i{1, 2, 3, 4}};

    auto stamp_ordered = build_pt_stamp(ordered, SizeT{99});
    auto stamp_reordered = build_pt_stamp(reordered, SizeT{99});
    CHECK(stamp_ordered.layout_hash == stamp_reordered.layout_hash);
    CHECK(stamp_ordered.content_hash != stamp_reordered.content_hash);

    auto source_order = build_two_source_stamp(false);
    auto source_order_swapped = build_two_source_stamp(true);
    CHECK(source_order.layout_hash != source_order_swapped.layout_hash);
    CHECK(source_order.content_hash != source_order_swapped.content_hash);
}

TEST_CASE("cuda_mixed_socu_contact_topology_change_rebuilds_program_only",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    if(!has_cuda_device())
        SKIP("no CUDA device is available for SOCU contact topology stamp tests");

    const std::vector<Vector4i> topology_a = {Vector4i{1, 2, 3, 4}};
    const std::vector<Vector4i> topology_b = {Vector4i{1, 2, 3, 5}};

    SocuContactTopologyStampCache stamp_cache;
    const auto stamp_a =
        stamp_cache.update(build_pt_stamp(topology_a, SizeT{707}));
    const auto stamp_b =
        stamp_cache.update(build_pt_stamp(topology_b, SizeT{707}));

    SocuContactPlanCacheState plan_cache;
    auto decision = plan_cache.update(plan_key_from_stamp(stamp_a));
    CHECK(!decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    decision = plan_cache.update(plan_key_from_stamp(stamp_b));
    CHECK(decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());
}
