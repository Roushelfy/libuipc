#include <tcl/case_spec.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

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

void store_matrix(const Mat16& src, std::vector<double>& dst, int batch_index)
{
    const size_t offset = static_cast<size_t>(batch_index) * 16 * 16;
    for(int c = 0; c < 16; ++c)
        for(int r = 0; r < 16; ++r)
            dst[offset + static_cast<size_t>(c) * 16 + r] = src(r, c);
}
}  // namespace

ContractionCaseData make_abd12_assemble_case(const std::string& name,
                                             int                batch_count,
                                             int                seed,
                                             double             condition_scale)
{
    ContractionCaseData out;
    out.spec.name            = name;
    out.spec.op_kind         = OpKind::Abd12Assemble;
    out.spec.batch_count     = batch_count;
    out.spec.seed            = seed;
    out.spec.condition_scale = condition_scale;
    out.spec.shape           = TensorShape{12, 12, 16, 16};

    const size_t batch_stride = 16 * 16;
    out.left.resize(batch_stride * static_cast<size_t>(batch_count));
    out.middle.resize(batch_stride * static_cast<size_t>(batch_count));
    out.aux.resize(batch_stride * static_cast<size_t>(batch_count));
    out.reference.resize(batch_stride * static_cast<size_t>(batch_count));

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

        // Keep the secondary term diagonal so the benchmark isolates the
        // contraction-plus-add pattern without introducing a second Jacobian.
        const Mat12 mass = a - j.transpose() * h * j;

        Mat16 j_pad    = Mat16::Zero();
        Mat16 h_pad    = Mat16::Zero();
        Mat16 mass_pad = Mat16::Zero();
        Mat16 a_pad    = Mat16::Zero();

        j_pad.block<9, 12>(0, 0)      = j;
        h_pad.block<9, 9>(0, 0)       = h;
        mass_pad.block<12, 12>(0, 0)  = mass;
        a_pad.block<12, 12>(0, 0)     = a;

        store_matrix(j_pad, out.left, batch);
        store_matrix(h_pad, out.middle, batch);
        store_matrix(mass_pad, out.aux, batch);
        store_matrix(a_pad, out.reference, batch);
    }

    return out;
}
}  // namespace uipc::tensor_core_lab
