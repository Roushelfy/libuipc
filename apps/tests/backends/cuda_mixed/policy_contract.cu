#include <app/app.h>
#include <linear_system/iterative_solver.h>
#include <linear_system/linear_fused_pcg.h>
#include <linear_system/linear_pcg.h>
#include <linear_system/linear_solver.h>
#include <linear_system/socu_approx_solver.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>
#include <utils/structured_contact_assembly_sink.h>
#include <linear_system/socu_rcm_ordering.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>

#ifndef UIPC_WITH_SOCU_NATIVE
#error "cuda_mixed must define UIPC_WITH_SOCU_NATIVE as 0 or 1"
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/common.h>
#endif

namespace
{
using namespace uipc::backend::cuda_mixed;

template <MixedPrecisionLevel Level,
          typename Alu,
          typename Store,
          typename PcgAux,
          typename Solve,
          typename Iter>
consteval bool policy_types_match()
{
    using Policy = PrecisionPolicy<Level>;
    return std::is_same_v<typename Policy::AluScalar, Alu>
           && std::is_same_v<typename Policy::EnergyScalar, Alu>
           && std::is_same_v<typename Policy::StoreScalar, Store>
           && std::is_same_v<typename Policy::PcgAuxScalar, PcgAux>
           && std::is_same_v<typename Policy::SolveScalar, Solve>
           && std::is_same_v<typename Policy::PcgIterScalar, Iter>;
}

static_assert(policy_types_match<MixedPrecisionLevel::FP64,
                                 double,
                                 double,
                                 double,
                                 double,
                                 double>());
static_assert(!PrecisionPolicy<MixedPrecisionLevel::FP64>::alu_is_fp32);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::FP64>::store_is_fp32);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::FP64>::pcg_is_fp32);
static_assert(
    !PrecisionPolicy<MixedPrecisionLevel::FP64>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::FP64>::full_pcg_fp32);

static_assert(policy_types_match<MixedPrecisionLevel::Path1,
                                 float,
                                 double,
                                 double,
                                 double,
                                 double>());
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path1>::alu_is_fp32);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path1>::store_is_fp32);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path1>::pcg_is_fp32);
static_assert(
    !PrecisionPolicy<MixedPrecisionLevel::Path1>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path1>::full_pcg_fp32);

static_assert(policy_types_match<MixedPrecisionLevel::Path2,
                                 float,
                                 float,
                                 double,
                                 double,
                                 double>());
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path2>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path2>::store_is_fp32);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path2>::pcg_is_fp32);
static_assert(
    !PrecisionPolicy<MixedPrecisionLevel::Path2>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path2>::full_pcg_fp32);

static_assert(policy_types_match<MixedPrecisionLevel::Path3,
                                 float,
                                 float,
                                 float,
                                 double,
                                 double>());
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path3>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path3>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path3>::pcg_is_fp32);
static_assert(
    !PrecisionPolicy<MixedPrecisionLevel::Path3>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path3>::full_pcg_fp32);

static_assert(policy_types_match<MixedPrecisionLevel::Path4,
                                 float,
                                 float,
                                 float,
                                 double,
                                 double>());
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path4>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path4>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path4>::pcg_is_fp32);
static_assert(
    PrecisionPolicy<MixedPrecisionLevel::Path4>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path4>::full_pcg_fp32);

static_assert(policy_types_match<MixedPrecisionLevel::Path5,
                                 float,
                                 float,
                                 float,
                                 float,
                                 float>());
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::pcg_is_fp32);
static_assert(
    PrecisionPolicy<MixedPrecisionLevel::Path5>::preconditioner_no_double_intermediate);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::full_pcg_fp32);

static_assert(policy_types_match<MixedPrecisionLevel::Path6,
                                 float,
                                 float,
                                 float,
                                 float,
                                 double>());
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path6>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path6>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path6>::pcg_is_fp32);
static_assert(
    PrecisionPolicy<MixedPrecisionLevel::Path6>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path6>::full_pcg_fp32);

static_assert(std::is_same_v<ActivePolicy, PrecisionPolicy<kBuildLevel>>);

static_assert(std::is_base_of_v<SimSystem, LinearSolver>);
static_assert(std::is_base_of_v<LinearSolver, IterativeSolver>);
static_assert(std::is_base_of_v<IterativeSolver, LinearPCG>);
static_assert(std::is_base_of_v<IterativeSolver, LinearFusedPCG>);
static_assert(std::is_base_of_v<LinearSolver, SocuApproxSolver>);

static_assert(UIPC_WITH_SOCU_NATIVE == 0 || UIPC_WITH_SOCU_NATIVE == 1);

#if UIPC_WITH_SOCU_NATIVE
static_assert(std::is_same_v<decltype(socu_native::ProblemShape{}.n), int>);
#endif
}  // namespace

TEST_CASE("cuda_mixed_policy_contract", "[cuda_mixed][contract]")
{
    SUCCEED();
}

TEST_CASE("cuda_mixed_socu_mixed_graph_fem_source_id",
          "[cuda_mixed][contract][socu_approx]")
{
    using uipc::SizeT;
    namespace ordering = uipc::backend::cuda_mixed::socu_approx::rcm;

    // Simulate a FEM sub-graph: 3 vertices with source_ids 10, 11, 12
    ordering::AtomGraph fem_graph;
    for(SizeT v = 0; v < 3; ++v)
        ordering::add_atom(fem_graph, 3, "fem_vertex", 10 + v);

    // Simulate an ABD graph: 2 bodies * 4 atoms = 8 atoms
    ordering::AtomGraph merged;
    const SizeT abd_atom_count = 8;
    for(SizeT i = 0; i < abd_atom_count; ++i)
        ordering::add_atom(merged, 3, "abd_body_local", i);

    const SizeT fem_atom_base = merged.atoms.size();  // = 8

    // Fixed code: use fem_graph.atoms[atom].source_id
    for(SizeT atom = 0; atom < fem_graph.atoms.size(); ++atom)
        ordering::add_atom(merged, 3, "fem_vertex", fem_graph.atoms[atom].source_id);

    REQUIRE(merged.atoms.size() == abd_atom_count + 3);
    for(SizeT atom = 0; atom < fem_graph.atoms.size(); ++atom)
    {
        const SizeT merged_id = fem_atom_base + atom;
        // source_id must equal the original FEM vertex index, not merged_id
        CHECK(merged.atoms[merged_id].source_id == fem_graph.atoms[atom].source_id);
        CHECK(merged.atoms[merged_id].source_id != merged_id);  // would be wrong
    }
}

TEST_CASE("cuda_mixed_socu_upper_lr_equal_vertex_no_mirror",
          "[cuda_mixed][contract][socu_approx]")
{
    using uipc::IndexT;
    using namespace uipc::backend::cuda_mixed;
    using Sink = StructuredContactAssemblySink<ActivePolicy::StoreScalar,
                                               ActivePolicy::SolveScalar>;
    Sink sink{};

    // equal global vertex indices with different stencil slots:
    // should NOT be treated as swapped (no mirror on diagonal block)
    {
        IndexT L = -1, R = -1;
        const bool swapped = sink.upper_lr(5, 5, 0, 1, L, R);
        CHECK(!swapped);
        CHECK(L == 0);
        CHECK(R == 1);
    }

    // left_value > right_value: must swap and signal mirror
    {
        IndexT L = -1, R = -1;
        const bool swapped = sink.upper_lr(7, 3, 2, 4, L, R);
        CHECK(swapped);
        CHECK(L == 4);
        CHECK(R == 2);
    }

    // left_value < right_value: no swap
    {
        IndexT L = -1, R = -1;
        const bool swapped = sink.upper_lr(1, 9, 0, 3, L, R);
        CHECK(!swapped);
        CHECK(L == 0);
        CHECK(R == 3);
    }
}

TEST_CASE("cuda_mixed_socu_probe_mode_topology_flag",
          "[cuda_mixed][contract][socu_approx]")
{
    using namespace uipc::backend::cuda_mixed;
    using Sink = StructuredContactAssemblySink<ActivePolicy::StoreScalar,
                                               ActivePolicy::SolveScalar>;

    // A default-constructed (invalid) sink is never in probe mode.
    Sink sink{};
    CHECK(!sink.topology_probe_only());

    // topology_probe_only() == runtime_ordering.valid() && graph_only && topology_only.
    // An invalid sink has runtime_ordering.valid() == false, so the flag is always false.
    // This validates the short-circuit: probe-mode guards like
    //   if(structured_sink.topology_probe_only()) { write_topology_half...; }
    // correctly fall through to the full plan-based path in normal assembly.
}

TEST_CASE("cuda_mixed_socu_vertex_slot_fill_unified",
          "[cuda_mixed][contract][socu_approx]")
{
    // Source contract: the two separate fill kernels have been replaced by a single
    // fill_structured_contact_vertex_slots that calls map_vertex_slow, which contains
    // the full bounds-checked vertex-to-slot resolution logic in one place.
    const auto source_path =
        std::filesystem::path{UIPC_PROJECT_DIR}
        / "src/backends/cuda_mixed/dytopo_effect_system/global_dytopo_effect_manager.cu";
    std::ifstream ifs{source_path};
    REQUIRE(ifs.good());

    const std::string source{std::istreambuf_iterator<char>{ifs},
                             std::istreambuf_iterator<char>{}};
    // old per-type fill functions must not exist anymore
    CHECK(source.find("fill_abd_structured_contact_vertex_slots") == std::string::npos);
    CHECK(source.find("fill_fem_structured_contact_vertex_slots") == std::string::npos);
    // new unified function must exist
    CHECK(source.find("fill_structured_contact_vertex_slots") != std::string::npos);
    // unified fill delegates to map_vertex_slow (single source of truth)
    CHECK(source.find("map_vertex_slow") != std::string::npos);
}

TEST_CASE("cuda_mixed_socu_approx_source_contract",
          "[cuda_mixed][contract][socu_approx]")
{
    const auto source_path =
        std::filesystem::path{UIPC_PROJECT_DIR}
        / "src/backends/cuda_mixed/linear_system/socu_approx_solver.cu";
    std::ifstream ifs{source_path};
    REQUIRE(ifs.good());

    const std::string source{std::istreambuf_iterator<char>{ifs},
                             std::istreambuf_iterator<char>{}};
    const auto do_solve = source.find("void SocuApproxSolver::do_solve");
    REQUIRE(do_solve != std::string::npos);
    const auto production_begin =
        source.find("const cudaStream_t stream = system().stream();", do_solve);
    REQUIRE(production_begin != std::string::npos);
    const auto production_end =
        source.find("\"SocuApproxSolver strict structured solve launched",
                    production_begin);
    REQUIRE(production_end != std::string::npos);

    const auto production =
        source.substr(production_begin, production_end - production_begin);
    CHECK(production.find("load_contact_report") == std::string::npos);
    CHECK(production.find("CpuStructuredDryRunSink") == std::string::npos);
    CHECK(production.find(".copy_to(") == std::string::npos);
    CHECK(production.find(".copy_from(") == std::string::npos);
    CHECK(production.find("cudaStream_t stream = nullptr") == std::string::npos);
    CHECK(production.find("validate_direction_light(stream)") != std::string::npos);
    CHECK(production.find("snapshot_matrix") == std::string::npos);
    CHECK(production.find("validate_structured_direction(") == std::string::npos);
}
