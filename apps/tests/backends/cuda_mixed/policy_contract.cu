#include <app/app.h>
#include <mixed_precision/policy.h>
#include <type_traits>

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
}  // namespace

TEST_CASE("cuda_mixed_policy_contract", "[cuda_mixed][contract]")
{
    SUCCEED();
}
