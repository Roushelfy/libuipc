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
    // Path1/Path3: ALU in fp32; Path2/Path4 rollback ALU to fp64 and focus on storage/PCG domains.
    using AluScalar =
        std::conditional_t<L == MixedPrecisionLevel::Path1 || L == MixedPrecisionLevel::Path3,
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
        std::conditional_t<L == MixedPrecisionLevel::Path4, float, double>;
    using PcgVector = muda::DeviceDenseVector<PcgAuxScalar>;

    using SolveScalar = double;

    static constexpr bool alu_is_fp32   = (L == MixedPrecisionLevel::Path1
                                           || L == MixedPrecisionLevel::Path3);
    static constexpr bool store_is_fp32 = (L == MixedPrecisionLevel::Path2
                                           || L == MixedPrecisionLevel::Path3
                                           || L == MixedPrecisionLevel::Path4);
    static constexpr bool pcg_is_fp32   = (L == MixedPrecisionLevel::Path4);
};

using ActivePolicy = PrecisionPolicy<kBuildLevel>;
}  // namespace uipc::backend::cuda_mixed
