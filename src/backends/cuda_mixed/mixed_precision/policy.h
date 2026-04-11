#pragma once
#include <mixed_precision/build_level.h>
#include <Eigen/Core>
#include <muda/ext/linear_system.h>
#include <type_traits>

namespace uipc::backend::cuda_mixed
{
template <MixedPrecisionLevel L>
struct PrecisionPolicy
{
    // Path1/Path3/Path5/Path6/Path7/Path8: ALU in fp32. Path2/Path4 keep ALU in fp64.
    using AluScalar =
        std::conditional_t<L == MixedPrecisionLevel::Path1
                               || L == MixedPrecisionLevel::Path3
                               || L == MixedPrecisionLevel::Path5
                               || L == MixedPrecisionLevel::Path6
                               || L == MixedPrecisionLevel::Path7
                               || L == MixedPrecisionLevel::Path8,
                           float,
                           double>;
    using AluMat3x3 = Eigen::Matrix<AluScalar, 3, 3>;
    using AluMat9x9 = Eigen::Matrix<AluScalar, 9, 9>;
    using AluVec12  = Eigen::Matrix<AluScalar, 12, 1>;
    using AluMat12x12 = Eigen::Matrix<AluScalar, 12, 12>;

    using StoreScalar =
        std::conditional_t<L == MixedPrecisionLevel::FP64 || L == MixedPrecisionLevel::Path1,
                           double,
                           float>;
    using TripletMatrix3 = muda::DeviceTripletMatrix<StoreScalar, 3>;
    using BCOOMatrix3    = muda::DeviceBCOOMatrix<StoreScalar, 3>;
    using GradientVec    = muda::DeviceDenseVector<StoreScalar>;

    using PcgAuxScalar =
        std::conditional_t<L == MixedPrecisionLevel::Path4
                               || L == MixedPrecisionLevel::Path5
                               || L == MixedPrecisionLevel::Path6
                               || L == MixedPrecisionLevel::Path7
                               || L == MixedPrecisionLevel::Path8,
                           float,
                           double>;
    using PcgVector = muda::DeviceDenseVector<PcgAuxScalar>;

    // Path7: full PCG fp32 (including solve vector x and iteration scalars).
    // Path8: diagnostic split path that keeps SolveScalar=float while restoring
    // PcgIterScalar=double to isolate iteration-scalar sensitivity.
    using SolveScalar =
        std::conditional_t<L == MixedPrecisionLevel::Path7
                               || L == MixedPrecisionLevel::Path8,
                           float,
                           double>;
    using PcgIterScalar =
        std::conditional_t<L == MixedPrecisionLevel::Path7, float, double>;

    static constexpr bool alu_is_fp32   = (L == MixedPrecisionLevel::Path1
                                           || L == MixedPrecisionLevel::Path3
                                           || L == MixedPrecisionLevel::Path5
                                           || L == MixedPrecisionLevel::Path6
                                           || L == MixedPrecisionLevel::Path7
                                           || L == MixedPrecisionLevel::Path8);
    static constexpr bool store_is_fp32 = (L == MixedPrecisionLevel::Path2
                                           || L == MixedPrecisionLevel::Path3
                                           || L == MixedPrecisionLevel::Path4
                                           || L == MixedPrecisionLevel::Path5
                                           || L == MixedPrecisionLevel::Path6
                                           || L == MixedPrecisionLevel::Path7
                                           || L == MixedPrecisionLevel::Path8);
    static constexpr bool pcg_is_fp32   = (L == MixedPrecisionLevel::Path4
                                           || L == MixedPrecisionLevel::Path5
                                           || L == MixedPrecisionLevel::Path6
                                           || L == MixedPrecisionLevel::Path7
                                           || L == MixedPrecisionLevel::Path8);
    static constexpr bool preconditioner_no_double_intermediate =
        (L == MixedPrecisionLevel::Path6 || L == MixedPrecisionLevel::Path7
         || L == MixedPrecisionLevel::Path8);
    static constexpr bool full_pcg_fp32 = (L == MixedPrecisionLevel::Path7);
};

using ActivePolicy = PrecisionPolicy<kBuildLevel>;

static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path5>::pcg_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path6>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path6>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path6>::pcg_is_fp32);
static_assert(
    PrecisionPolicy<MixedPrecisionLevel::Path6>::preconditioner_no_double_intermediate);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path7>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path7>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path7>::pcg_is_fp32);
static_assert(
    PrecisionPolicy<MixedPrecisionLevel::Path7>::preconditioner_no_double_intermediate);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path7>::full_pcg_fp32);
static_assert(std::is_same_v<PrecisionPolicy<MixedPrecisionLevel::Path7>::SolveScalar, float>);
static_assert(
    std::is_same_v<PrecisionPolicy<MixedPrecisionLevel::Path7>::PcgIterScalar, float>);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path8>::alu_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path8>::store_is_fp32);
static_assert(PrecisionPolicy<MixedPrecisionLevel::Path8>::pcg_is_fp32);
static_assert(
    PrecisionPolicy<MixedPrecisionLevel::Path8>::preconditioner_no_double_intermediate);
static_assert(!PrecisionPolicy<MixedPrecisionLevel::Path8>::full_pcg_fp32);
static_assert(std::is_same_v<PrecisionPolicy<MixedPrecisionLevel::Path8>::PcgAuxScalar, float>);
static_assert(std::is_same_v<PrecisionPolicy<MixedPrecisionLevel::Path8>::SolveScalar, float>);
static_assert(
    std::is_same_v<PrecisionPolicy<MixedPrecisionLevel::Path8>::PcgIterScalar, double>);
}  // namespace uipc::backend::cuda_mixed
