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
using Mat3  = Eigen::Matrix3d;
using Mat48 = Eigen::Matrix<double, 48, 48>;
using Vec48 = Eigen::Matrix<double, 48, 1>;

Mat3 random_spd3(std::mt19937& rng)
{
    Eigen::Matrix3d raw;
    for(int col = 0; col < 3; ++col)
        for(int row = 0; row < 3; ++row)
            raw(row, col) = normal_real(rng);

    return raw.transpose() * raw + 0.1 * Mat3::Identity();
}

Mat48 condition_matrix(const Mat48& base, double condition_scale)
{
    Eigen::SelfAdjointEigenSolver<Mat48> es((base + base.transpose()) * 0.5);
    Eigen::Matrix<double, 48, 1> evals;
    for(int i = 0; i < 48; ++i)
    {
        const double t = static_cast<double>(i) / 47.0;
        evals(i)       = std::pow(condition_scale, t);
    }
    return es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}
}  // namespace

SpdCaseData make_mas48_case(const std::string& name,
                            int                batch_count,
                            int                seed,
                            double             condition_scale)
{
    SpdCaseData out;
    out.spec.name            = name;
    out.spec.op_kind         = OpKind::Mas48Factorize;
    out.spec.batch_count     = batch_count;
    out.spec.seed            = seed;
    out.spec.condition_scale = condition_scale;
    out.spec.shape           = TensorShape{48, 48, 48, 48};

    zero_square_storage(out.matrix, 48, batch_count);
    zero_square_storage(out.inverse_reference, 48, batch_count);
    zero_vector_storage(out.rhs, 48, batch_count);
    zero_vector_storage(out.solution_reference, 48, batch_count);

    for(int batch = 0; batch < batch_count; ++batch)
    {
        auto rng = make_rng(seed, batch, 48);

        Mat48 base = Mat48::Zero();

        for(int node = 0; node < 16; ++node)
        {
            const double mass = 2.0 + 0.25 * uniform_real(rng, 0.0, 1.0);
            base.template block<3, 3>(3 * node, 3 * node) += mass * Mat3::Identity();
        }

        for(int edge = 0; edge < 16; ++edge)
        {
            const int i = edge;
            const int j = (edge + 1) % 16;
            const int k = (edge + 5) % 16;

            const Mat3 ke_ring = random_spd3(rng);
            const Mat3 ke_skip = 0.35 * random_spd3(rng);

            base.template block<3, 3>(3 * i, 3 * i) += ke_ring + ke_skip;
            base.template block<3, 3>(3 * j, 3 * j) += ke_ring;
            base.template block<3, 3>(3 * k, 3 * k) += ke_skip;
            base.template block<3, 3>(3 * i, 3 * j) -= ke_ring;
            base.template block<3, 3>(3 * j, 3 * i) -= ke_ring;
            base.template block<3, 3>(3 * i, 3 * k) -= ke_skip;
            base.template block<3, 3>(3 * k, 3 * i) -= ke_skip;
        }

        Mat48 a = condition_matrix(base + 1.0e-2 * Mat48::Identity(), condition_scale);
        a += 1.0e-3 * Mat48::Identity();

        Vec48 rhs;
        for(int i = 0; i < 48; ++i)
            rhs(i) = normal_real(rng);

        const Eigen::LLT<Mat48> llt(a);
        const Mat48             inv = llt.solve(Mat48::Identity());
        const Vec48             sol = llt.solve(rhs);

        store_square_padded(a, 48, 48, 48, 48, out.matrix, batch);
        store_square_padded(inv, 48, 48, 48, 48, out.inverse_reference, batch);
        store_vector_padded(rhs, 48, 48, out.rhs, batch);
        store_vector_padded(sol, 48, 48, out.solution_reference, batch);
    }

    return out;
}
}  // namespace uipc::tensor_core_lab
