#include <app/app.h>
#include <linear_system/linear_solver.h>
#include <linear_system/socu_contact_assembly_plan.h>
#include <linear_system/socu_contact_plan_types.h>
#include <linear_system/socu_approx_report.h>
#include <linear_system/socu_approx_solver.h>
#include <linear_system/socu_rcm_ordering.h>
#include <mixed_precision/policy.h>
#include <uipc/common/json.h>
#include <utils/structured_contact_assembly_sink.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>

#ifndef UIPC_WITH_SOCU_NATIVE
#error "cuda_mixed_socu must define UIPC_WITH_SOCU_NATIVE as 0 or 1"
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/common.h>
#endif

namespace
{
using namespace uipc::backend::cuda_mixed;
using uipc::SizeT;

static_assert(std::is_base_of_v<LinearSolver, SocuApproxSolver>);
static_assert(UIPC_WITH_SOCU_NATIVE == 0 || UIPC_WITH_SOCU_NATIVE == 1);

#if UIPC_WITH_SOCU_NATIVE
static_assert(std::is_same_v<decltype(socu_native::ProblemShape{}.n), int>);
#endif
}  // namespace

TEST_CASE("cuda_mixed_socu_policy_contract", "[cuda_mixed_socu][contract]")
{
    SUCCEED();
}

TEST_CASE("cuda_mixed_socu_report_native_contact_plan_defaults",
          "[cuda_mixed_socu][contract][socu_approx]")
{
    using uipc::Json;
    using uipc::SizeT;
    using uipc::backend::cuda_mixed::SocuApproxSolveReport;
    using uipc::backend::cuda_mixed::write_solve_report;

    const auto report_path =
        std::filesystem::temp_directory_path()
        / "uipc_socu_report_native_contact_plan_defaults.json";
    std::filesystem::remove(report_path);

    SocuApproxSolveReport report;
    report.report_path = report_path.string();
    write_solve_report(report);

    std::ifstream ifs{report_path};
    REQUIRE(ifs.good());
    const Json json = Json::parse(ifs);

    const auto& timing = json.at("timing");
    CHECK(timing.at("native_contact_plan_enabled").get<bool>() == false);
    CHECK(timing.at("native_contact_plan_executor_enabled").get<bool>() == false);
    CHECK(timing.at("native_contact_hot_reduce_enabled").get<bool>() == false);
    CHECK(timing.at("native_contact_scalar_diag_compat_enabled").get<bool>() == false);
    CHECK(timing.at("native_contact_hot_reduce_strategy").get<std::string>()
          == "off");
    CHECK(timing.at("native_contact_plan_build_ms").get<double>() == 0.0);
    CHECK(timing.at("native_contact_side_plan_build_ms").get<double>() == 0.0);
    CHECK(timing.at("native_contact_program_plan_build_ms").get<double>() == 0.0);
    CHECK(timing.at("native_contact_side_coverage_refresh_ms").get<double>() == 0.0);
    CHECK(timing.at("native_contact_numeric_ms").get<double>() == 0.0);
    CHECK(timing.at("native_contact_hot_reduce_ms").get<double>() == 0.0);

    const auto& contact = json.at("contact");
    CHECK(contact.at("native_contact_plan_cache_hit").get<bool>() == false);
    CHECK(contact.at("native_contact_side_plan_cache_hit").get<bool>() == false);
    CHECK(contact.at("native_contact_program_plan_cache_hit").get<bool>() == false);
    CHECK(contact.at("native_contact_side_coverage_cache_hit").get<bool>() == false);
    CHECK(contact.at("native_contact_active_side_set_changed").get<bool>() == false);
    CHECK(contact.at("native_contact_side_coverage_mode").get<std::string>()
          == "off");
    CHECK(contact.at("native_contact_source_id_validation_status").get<std::string>()
          == "not_run");
    for(const char* field : {"native_contact_plan_rebuild_count",
                             "native_contact_side_plan_rebuild_count",
                             "native_contact_program_plan_rebuild_count",
                             "native_contact_side_coverage_refresh_count",
                             "native_contact_side_coverage_fill_count",
                             "native_contact_active_side_set_changed_count",
                             "native_contact_active_side_vertex_count",
                             "native_contact_side_count",
                             "native_contact_lane_count",
                             "native_contact_source_count",
                             "native_contact_program_count",
                             "native_contact_source_to_program_count",
                             "native_contact_valid_program_map_count",
                             "native_contact_missing_program_map_count",
                             "native_contact_invalid_program_map_count",
                             "native_contact_dropped_program_map_count",
                             "native_contact_skipped_program_map_count",
                             "native_contact_mixed_rejected_program_map_count",
                             "native_contact_task_count",
                             "native_contact_bucket_count",
                             "native_contact_exact_program_count",
                             "native_contact_diag_program_count",
                             "native_contact_diag_lump_program_count",
                             "native_contact_drop_program_count",
                             "native_contact_skipped_program_count",
                             "native_contact_mixed_rejected_program_count",
                             "native_contact_diag_block_task_count",
                             "native_contact_diag_scalar_task_count",
                             "native_contact_lump_scalar_task_count",
                             "native_contact_hot_diag_block_count",
                             "native_contact_hot_offdiag_block_count"})
    {
        CAPTURE(field);
        CHECK(contact.at(field).get<SizeT>() == SizeT{0});
    }

    std::filesystem::remove(report_path);
}

TEST_CASE("cuda_mixed_socu_report_native_contact_plan_stats_mapping",
          "[cuda_mixed_socu][contract][socu_approx][m2]")
{
    using uipc::Json;

    SocuApproxSolveReport report;
    SocuContactPlanStats side_stats;
    SocuContactPlanStats program_stats;
    side_stats.side_count = 4;
    side_stats.lane_count = 21;
    side_stats.coverage_mode = SocuVertexSideCoverageMode::ActiveSetTemporary;
    side_stats.active_side_vertex_count = 4;
    program_stats.source_count = 3;
    program_stats.source_id_validation_status =
        SocuContactSourceIdValidationStatus::ValidDense;
    program_stats.program_count = 7;
    program_stats.source_to_program_count = 7;
    program_stats.valid_program_map_count = 4;
    program_stats.missing_program_map_count = 0;
    program_stats.invalid_program_map_count = 0;
    program_stats.dropped_program_map_count = 2;
    program_stats.skipped_program_map_count = 1;
    program_stats.mixed_rejected_program_map_count = 0;
    program_stats.task_count = 19;
    program_stats.bucket_count = 5;
    program_stats.exact_program_count = 2;
    program_stats.diag_program_count = 1;
    program_stats.diag_lump_program_count = 1;
    program_stats.drop_program_count = 2;
    program_stats.skipped_program_count = 1;
    program_stats.mixed_rejected_program_count = 0;
    program_stats.diag_block_task_count = 3;
    program_stats.diag_scalar_task_count = 4;
    program_stats.lump_scalar_task_count = 5;
    program_stats.hot_diag_block_count = 6;
    program_stats.hot_offdiag_block_count = 7;
    apply_native_contact_plan_stats(report, side_stats, program_stats);

    CHECK(report.native_contact_side_count == 4);
    CHECK(report.native_contact_lane_count == 21);
    CHECK(report.native_contact_side_coverage_mode == "active_set_temporary");
    CHECK(report.native_contact_active_side_vertex_count == 4);
    CHECK(report.native_contact_source_id_validation_status == "valid_dense");
    CHECK(report.native_contact_source_count == 3);
    CHECK(report.native_contact_program_count == 7);
    CHECK(report.native_contact_source_to_program_count == 7);
    CHECK(report.native_contact_valid_program_map_count == 4);
    CHECK(report.native_contact_missing_program_map_count == 0);
    CHECK(report.native_contact_invalid_program_map_count == 0);
    CHECK(report.native_contact_dropped_program_map_count == 2);
    CHECK(report.native_contact_skipped_program_map_count == 1);
    CHECK(report.native_contact_mixed_rejected_program_map_count == 0);
    CHECK(report.native_contact_task_count == 19);
    CHECK(report.native_contact_bucket_count == 5);
    CHECK(report.native_contact_exact_program_count == 2);
    CHECK(report.native_contact_diag_program_count == 1);
    CHECK(report.native_contact_diag_lump_program_count == 1);
    CHECK(report.native_contact_drop_program_count == 2);
    CHECK(report.native_contact_skipped_program_count == 1);
    CHECK(report.native_contact_diag_block_task_count == 3);
    CHECK(report.native_contact_diag_scalar_task_count == 4);
    CHECK(report.native_contact_lump_scalar_task_count == 5);
    CHECK(report.native_contact_hot_diag_block_count == 6);
    CHECK(report.native_contact_hot_offdiag_block_count == 7);

    SocuContactPlanCacheState cache;
    SocuAssemblyPlanKey key;
    key.ordering_epoch = 1;
    key.native_descriptor_epoch = 2;
    key.contact_topology_epoch = 3;
    key.contact_layout_hash = 4;
    key.contact_content_hash = 5;
    key.fixed_mapping_epoch = 6;
    key.vertex_projection_epoch = 7;
    key.horizon = 8;
    key.block_size = 16;
    SizeT rebuild_count = 0;

    auto decision = cache.update(key);
    if(!(decision.side_plan_hit() && decision.contact_program_hit()))
        ++rebuild_count;
    SizeT side_rebuild_count = decision.side_plan_hit() ? 0 : 1;
    SizeT program_rebuild_count = decision.contact_program_hit() ? 0 : 1;
    apply_native_contact_plan_cache_decision(report,
                                             decision,
                                             rebuild_count,
                                             side_rebuild_count,
                                             program_rebuild_count);
    CHECK(!report.native_contact_plan_cache_hit);
    CHECK(report.native_contact_plan_rebuild_count == 1);
    CHECK(!report.native_contact_side_plan_cache_hit);
    CHECK(!report.native_contact_program_plan_cache_hit);
    CHECK(report.native_contact_side_plan_rebuild_count == 1);
    CHECK(report.native_contact_program_plan_rebuild_count == 1);

    auto topology_changed = key;
    ++topology_changed.contact_topology_epoch;
    decision = cache.update(topology_changed);
    if(!(decision.side_plan_hit() && decision.contact_program_hit()))
        ++rebuild_count;
    if(!decision.side_plan_hit())
        ++side_rebuild_count;
    if(!decision.contact_program_hit())
        ++program_rebuild_count;
    apply_native_contact_plan_cache_decision(report,
                                             decision,
                                             rebuild_count,
                                             side_rebuild_count,
                                             program_rebuild_count);
    CHECK(decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());
    CHECK(!report.native_contact_plan_cache_hit);
    CHECK(report.native_contact_plan_rebuild_count == 2);
    CHECK(report.native_contact_side_plan_cache_hit);
    CHECK(!report.native_contact_program_plan_cache_hit);
    CHECK(report.native_contact_side_plan_rebuild_count == 1);
    CHECK(report.native_contact_program_plan_rebuild_count == 2);

    decision = cache.update(topology_changed);
    if(!(decision.side_plan_hit() && decision.contact_program_hit()))
        ++rebuild_count;
    if(!decision.side_plan_hit())
        ++side_rebuild_count;
    if(!decision.contact_program_hit())
        ++program_rebuild_count;
    apply_native_contact_plan_cache_decision(report,
                                             decision,
                                             rebuild_count,
                                             side_rebuild_count,
                                             program_rebuild_count);
    CHECK(report.native_contact_plan_cache_hit);
    CHECK(report.native_contact_plan_rebuild_count == 2);
    CHECK(report.native_contact_side_plan_cache_hit);
    CHECK(report.native_contact_program_plan_cache_hit);

    const auto report_path =
        std::filesystem::temp_directory_path()
        / "uipc_socu_report_native_contact_plan_stats.json";
    std::filesystem::remove(report_path);
    report.report_path = report_path.string();
    write_solve_report(report);

    std::ifstream ifs{report_path};
    REQUIRE(ifs.good());
    const Json json = Json::parse(ifs);
    const auto& contact = json.at("contact");
    CHECK(contact.at("native_contact_side_count").get<SizeT>() == 4);
    CHECK(contact.at("native_contact_side_coverage_mode").get<std::string>()
          == "active_set_temporary");
    CHECK(contact.at("native_contact_active_side_vertex_count").get<SizeT>() == 4);
    CHECK(contact.at("native_contact_source_count").get<SizeT>() == 3);
    CHECK(contact.at("native_contact_source_id_validation_status").get<std::string>()
          == "valid_dense");
    CHECK(contact.at("native_contact_program_count").get<SizeT>() == 7);
    CHECK(contact.at("native_contact_valid_program_map_count").get<SizeT>() == 4);
    CHECK(contact.at("native_contact_dropped_program_map_count").get<SizeT>() == 2);
    CHECK(contact.at("native_contact_bucket_count").get<SizeT>() == 5);
    CHECK(contact.at("native_contact_exact_program_count").get<SizeT>() == 2);
    CHECK(contact.at("native_contact_drop_program_count").get<SizeT>() == 2);
    CHECK(contact.at("native_contact_plan_cache_hit").get<bool>());
    CHECK(contact.at("native_contact_plan_rebuild_count").get<SizeT>() == 2);
    CHECK(contact.at("native_contact_side_plan_cache_hit").get<bool>());
    CHECK(contact.at("native_contact_program_plan_cache_hit").get<bool>());
    CHECK(contact.at("native_contact_side_plan_rebuild_count").get<SizeT>() == 1);
    CHECK(contact.at("native_contact_program_plan_rebuild_count").get<SizeT>() == 2);

    std::filesystem::remove(report_path);
}

TEST_CASE("cuda_mixed_socu_mixed_graph_fem_source_id",
          "[cuda_mixed_socu][contract][socu_approx]")
{
    using uipc::SizeT;
    namespace ordering = uipc::backend::cuda_mixed::socu_approx::rcm;

    ordering::AtomGraph fem_graph;
    for(SizeT v = 0; v < 3; ++v)
        ordering::add_atom(fem_graph, 3, "fem_vertex", 10 + v);

    ordering::AtomGraph merged;
    const SizeT         abd_atom_count = 8;
    for(SizeT i = 0; i < abd_atom_count; ++i)
        ordering::add_atom(merged, 3, "abd_body_local", i);

    const SizeT fem_atom_base = merged.atoms.size();
    for(SizeT atom = 0; atom < fem_graph.atoms.size(); ++atom)
        ordering::add_atom(merged, 3, "fem_vertex", fem_graph.atoms[atom].source_id);

    REQUIRE(merged.atoms.size() == abd_atom_count + 3);
    for(SizeT atom = 0; atom < fem_graph.atoms.size(); ++atom)
    {
        const SizeT merged_id = fem_atom_base + atom;
        CHECK(merged.atoms[merged_id].source_id == fem_graph.atoms[atom].source_id);
        CHECK(merged.atoms[merged_id].source_id != merged_id);
    }
}

TEST_CASE("cuda_mixed_socu_upper_lr_equal_vertex_no_mirror",
          "[cuda_mixed_socu][contract][socu_approx]")
{
    using uipc::IndexT;
    using Sink = StructuredContactAssemblySink<ActivePolicy::StoreScalar,
                                               ActivePolicy::SolveScalar>;
    Sink sink{};

    {
        IndexT     L       = -1;
        IndexT     R       = -1;
        const bool swapped = sink.upper_lr(5, 5, 0, 1, L, R);
        CHECK(!swapped);
        CHECK(L == 0);
        CHECK(R == 1);
    }

    {
        IndexT     L       = -1;
        IndexT     R       = -1;
        const bool swapped = sink.upper_lr(7, 3, 2, 4, L, R);
        CHECK(swapped);
        CHECK(L == 4);
        CHECK(R == 2);
    }

    {
        IndexT     L       = -1;
        IndexT     R       = -1;
        const bool swapped = sink.upper_lr(1, 9, 0, 3, L, R);
        CHECK(!swapped);
        CHECK(L == 0);
        CHECK(R == 3);
    }
}

TEST_CASE("cuda_mixed_socu_probe_mode_topology_flag",
          "[cuda_mixed_socu][contract][socu_approx]")
{
    using Sink = StructuredContactAssemblySink<ActivePolicy::StoreScalar,
                                               ActivePolicy::SolveScalar>;

    Sink sink{};
    CHECK(!sink.topology_probe_only());
}

TEST_CASE("cuda_mixed_socu_contact_topology_stamp_contract",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    SocuContactTopologySource source;
    source.reporter_id = 0;
    source.source_id = 0;
    source.family = SocuContactSourceFamily::SimplexNormalPT;
    source.contact_count = 2;
    source.layout_token = 17;
    source.content_hash = 101;

    auto same_count_different_vertices = source;
    same_count_different_vertices.content_hash = 202;

    CHECK(socu_contact_source_layout_hash(source)
          == socu_contact_source_layout_hash(same_count_different_vertices));
    CHECK(socu_contact_source_content_hash(source)
          != socu_contact_source_content_hash(same_count_different_vertices));

    auto geometry_only = source;
    CHECK(socu_contact_source_layout_hash(source)
          == socu_contact_source_layout_hash(geometry_only));
    CHECK(socu_contact_source_content_hash(source)
          == socu_contact_source_content_hash(geometry_only));

    auto different_storage_layout = source;
    different_storage_layout.layout_token = 18;
    CHECK(socu_contact_source_layout_hash(source)
          != socu_contact_source_layout_hash(different_storage_layout));
}

TEST_CASE("cuda_mixed_socu_contact_source_id_dense_contract",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    std::array<SocuContactTopologySource, 3> dense{};
    for(SizeT i = 0; i < dense.size(); ++i)
        dense[i].source_id = i;
    CHECK(socu_contact_source_ids_dense(dense));
    CHECK(socu_validate_contact_source_ids_dense(dense)
          == SocuContactSourceIdValidationStatus::ValidDense);

    auto non_dense = dense;
    non_dense[1].source_id = 2;
    non_dense[2].source_id = 1;
    CHECK(!socu_contact_source_ids_dense(non_dense));
    CHECK(socu_validate_contact_source_ids_dense(non_dense)
          == SocuContactSourceIdValidationStatus::NonDense);

    auto duplicate = dense;
    duplicate[2].source_id = 1;
    CHECK(!socu_contact_source_ids_dense(duplicate));
    CHECK(socu_validate_contact_source_ids_dense(duplicate)
          == SocuContactSourceIdValidationStatus::Duplicate);

    auto out_of_range = dense;
    out_of_range[2].source_id = 7;
    CHECK(!socu_contact_source_ids_dense(out_of_range));
    CHECK(socu_validate_contact_source_ids_dense(out_of_range)
          == SocuContactSourceIdValidationStatus::OutOfRange);
}

TEST_CASE("cuda_mixed_socu_contact_plan_key_invalidates_on_symbolic_inputs",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    SocuAssemblyPlanKey base;
    base.ordering_epoch = 7;
    base.native_descriptor_epoch = 11;
    base.contact_topology_epoch = 13;
    base.contact_layout_hash = 17;
    base.contact_content_hash = 19;
    base.fixed_mapping_epoch = 23;
    base.vertex_projection_epoch = 29;
    base.horizon = 31;
    base.block_size = 12;
    base.offband_policy = StructuredContactOffbandPolicy::Diag;
    base.scalar_diag_fallback_compatibility = true;

    const auto base_hash = socu_contact_plan_key_hash(base);

    auto changed = base;
    changed.ordering_epoch++;
    CHECK(changed != base);
    CHECK(socu_contact_plan_key_hash(changed) != base_hash);

    changed = base;
    changed.offband_policy = StructuredContactOffbandPolicy::DiagLump;
    CHECK(changed != base);
    CHECK(socu_contact_plan_key_hash(changed) != base_hash);

    changed = base;
    changed.fixed_mapping_epoch++;
    CHECK(changed != base);
    CHECK(socu_contact_plan_key_hash(changed) != base_hash);

    changed = base;
    changed.vertex_projection_epoch++;
    CHECK(changed != base);
    CHECK(socu_contact_plan_key_hash(changed) != base_hash);

    changed = base;
    changed.contact_content_hash++;
    CHECK(changed != base);
    CHECK(socu_contact_plan_key_hash(changed) != base_hash);
}

TEST_CASE("cuda_mixed_socu_contact_plan_cache_split_layers",
          "[cuda_mixed_socu][contract][socu_approx][m1]")
{
    SocuAssemblyPlanKey base;
    base.ordering_epoch = 7;
    base.native_descriptor_epoch = 11;
    base.contact_topology_epoch = 13;
    base.contact_layout_hash = 17;
    base.contact_content_hash = 19;
    base.fixed_mapping_epoch = 23;
    base.vertex_projection_epoch = 29;
    base.horizon = 31;
    base.block_size = 12;
    base.offband_policy = StructuredContactOffbandPolicy::Drop;

    SocuContactPlanCacheState cache;

    auto decision = cache.update(base);
    CHECK(decision.cold_start);
    CHECK(!decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    decision = cache.update(base);
    CHECK(!decision.cold_start);
    CHECK(decision.side_plan_hit());
    CHECK(decision.contact_program_hit());

    auto topology_changed = base;
    topology_changed.contact_content_hash++;
    decision = cache.update(topology_changed);
    CHECK(decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto offband_changed = topology_changed;
    offband_changed.offband_policy = StructuredContactOffbandPolicy::Diag;
    decision = cache.update(offband_changed);
    CHECK(decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto scalar_diag_changed = offband_changed;
    scalar_diag_changed.scalar_diag_fallback_compatibility = true;
    decision = cache.update(scalar_diag_changed);
    CHECK(decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto ordering_changed = scalar_diag_changed;
    ordering_changed.ordering_epoch++;
    decision = cache.update(ordering_changed);
    CHECK(!decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto descriptor_changed = ordering_changed;
    descriptor_changed.native_descriptor_epoch++;
    decision = cache.update(descriptor_changed);
    CHECK(!decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto fixed_mapping_changed = descriptor_changed;
    fixed_mapping_changed.fixed_mapping_epoch++;
    decision = cache.update(fixed_mapping_changed);
    CHECK(!decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto projection_changed = fixed_mapping_changed;
    projection_changed.vertex_projection_epoch++;
    decision = cache.update(projection_changed);
    CHECK(!decision.side_plan_hit());
    CHECK(!decision.contact_program_hit());

    auto geometry_only = projection_changed;
    decision = cache.update(geometry_only);
    CHECK(decision.side_plan_hit());
    CHECK(decision.contact_program_hit());
}
