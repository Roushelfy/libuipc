#include <tcl/case_spec.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "../common/padding.h"
#include "../common/rng.h"

namespace uipc::tensor_core_lab
{
namespace
{
using Mat3   = Eigen::Matrix3d;
using Mat9   = Eigen::Matrix<double, 9, 9>;
using Mat12  = Eigen::Matrix<double, 12, 12>;
using Mat16  = Eigen::Matrix<double, 16, 16>;
using Mat9x12 = Eigen::Matrix<double, 9, 12>;
using Vec12  = Eigen::Matrix<double, 12, 1>;
using Vec16  = Eigen::Matrix<double, 16, 1>;

Mat9 random_spd9(std::mt19937& rng)
{
    Mat9 raw;
    for(int col = 0; col < 9; ++col)
        for(int row = 0; row < 9; ++row)
            raw(row, col) = normal_real(rng);
    return raw.transpose() * raw + 0.05 * Mat9::Identity();
}

Mat12 condition_matrix(const Mat12& base, double condition_scale)
{
    Eigen::SelfAdjointEigenSolver<Mat12> es((base + base.transpose()) * 0.5);
    Eigen::Matrix<double, 12, 1> evals;
    for(int i = 0; i < 12; ++i)
    {
        const double t = static_cast<double>(i) / 11.0;
        evals(i)       = std::pow(condition_scale, t);
    }
    return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}

Eigen::Matrix<double, 3, 3> skew(const Eigen::Vector3d& p)
{
    Eigen::Matrix<double, 3, 3> s;
    s << 0.0, -p.z(), p.y(), p.z(), 0.0, -p.x(), -p.y(), p.x(), 0.0;
    return s;
}
}  // namespace

SpdCaseData make_abd12_case(const std::string& name,
                            int                batch_count,
                            int                seed,
                            double             condition_scale)
{
    SpdCaseData out;
    out.spec.name            = name;
    out.spec.op_kind         = OpKind::Abd12Factorize;
    out.spec.batch_count     = batch_count;
    out.spec.seed            = seed;
    out.spec.condition_scale = condition_scale;
    out.spec.shape           = TensorShape{12, 12, 16, 16};

    zero_square_storage(out.matrix, 16, batch_count);
    zero_square_storage(out.inverse_reference, 16, batch_count);
    zero_vector_storage(out.rhs, 16, batch_count);
    zero_vector_storage(out.solution_reference, 16, batch_count);

    for(int batch = 0; batch < batch_count; ++batch)
    {
        auto rng = make_rng(seed, batch, 12);

        Mat9x12 j = Mat9x12::Zero();
        for(int block = 0; block < 3; ++block)
        {
            Eigen::Vector3d p(uniform_real(rng, -0.7, 0.7),
                              uniform_real(rng, -0.7, 0.7),
                              uniform_real(rng, -0.7, 0.7));

            j.template block<3, 3>(3 * block, 0) = Mat3::Identity();
            j.template block<3, 3>(3 * block, 3) = -skew(p);
            j.template block<3, 3>(3 * block, 6) = 0.65 * Mat3::Identity();
            j.template block<3, 3>(3 * block, 9) = -0.65 * skew(p);
        }

        const Mat9  h = random_spd9(rng);
        const Mat12 m = (0.35 + 0.1 * uniform_real(rng, 0.0, 1.0)) * Mat12::Identity();
        Mat12       a = j.transpose() * h * j + m;
        a             = condition_matrix(a + 1.0e-2 * Mat12::Identity(), condition_scale);
        a += 1.0e-3 * Mat12::Identity();

        Vec12 rhs12;
        for(int i = 0; i < 12; ++i)
            rhs12(i) = normal_real(rng);

        const Eigen::LLT<Mat12> llt(a);
        const Mat12             inv12 = llt.solve(Mat12::Identity());
        const Vec12             sol12 = llt.solve(rhs12);

        Mat16 a_pad   = Mat16::Identity();
        Mat16 inv_pad = Mat16::Identity();
        a_pad.template block<12, 12>(0, 0)   = a;
        inv_pad.template block<12, 12>(0, 0) = inv12;

        Vec16 rhs16 = Vec16::Zero();
        Vec16 sol16 = Vec16::Zero();
        rhs16.template head<12>() = rhs12;
        sol16.template head<12>() = sol12;

        store_square_padded(a_pad, 16, 16, 16, 16, out.matrix, batch);
        store_square_padded(inv_pad, 16, 16, 16, 16, out.inverse_reference, batch);
        store_vector_padded(rhs16, 16, 16, out.rhs, batch);
        store_vector_padded(sol16, 16, 16, out.solution_reference, batch);
    }

    return out;
}
}  // namespace uipc::tensor_core_lab
