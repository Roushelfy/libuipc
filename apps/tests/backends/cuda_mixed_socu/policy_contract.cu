#include <app/app.h>
#include <linear_system/linear_solver.h>
#include <linear_system/socu_approx_solver.h>
#include <linear_system/socu_rcm_ordering.h>
#include <mixed_precision/policy.h>
#include <utils/structured_contact_assembly_sink.h>
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
