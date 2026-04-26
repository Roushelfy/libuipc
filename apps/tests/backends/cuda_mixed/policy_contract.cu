#include <app/app.h>
#include <linear_system/iterative_solver.h>
#include <linear_system/linear_fused_pcg.h>
#include <linear_system/linear_pcg.h>
#include <linear_system/linear_solver.h>
#include <linear_system/socu_approx_solver.h>
#include <mixed_precision/policy.h>
#include <utils/assembly_sink.h>
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
        source.find("logger::info(\"SocuApproxSolver strict structured solve launched",
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
