#include <tcl/case_spec.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include <array>
#include <cmath>
#include <random>

namespace uipc::tensor_core_lab
{
namespace
{
using Vec3    = Eigen::Vector3d;
using Mat3    = Eigen::Matrix3d;
using Mat9    = Eigen::Matrix<double, 9, 9>;
using Mat12   = Eigen::Matrix<double, 12, 12>;
using Mat9x12 = Eigen::Matrix<double, 9, 12>;
using Mat16   = Eigen::Matrix<double, 16, 16>;

Vec3 random_vec(std::mt19937& rng, double scale)
{
    std::uniform_real_distribution<double> dist(-scale, scale);
    return {dist(rng), dist(rng), dist(rng)};
}

Mat9 random_spd9(std::mt19937& rng, double condition_scale)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::Matrix<double, 9, 9> raw;
    for(int c = 0; c < 9; ++c)
        for(int r = 0; r < 9; ++r)
            raw(r, c) = dist(rng);

    Eigen::HouseholderQR<Eigen::Matrix<double, 9, 9>> qr(raw);
    const Eigen::Matrix<double, 9, 9> q =
        qr.householderQ() * Eigen::Matrix<double, 9, 9>::Identity();

    Eigen::Matrix<double, 9, 1> diag;
    for(int i = 0; i < 9; ++i)
    {
        const double t = static_cast<double>(i) / 8.0;
        diag(i)        = std::pow(condition_scale, t);
    }

    return q * diag.asDiagonal() * q.transpose() + 1.0e-2 * Mat9::Identity();
}

Mat9x12 tetra_dfdx(const Mat3& dm_inv)
{
    Mat9x12 dfdx = Mat9x12::Zero();

    const Vec3 g1 = dm_inv.col(0);
    const Vec3 g2 = dm_inv.col(1);
    const Vec3 g3 = dm_inv.col(2);
    const Vec3 g0 = -(g1 + g2 + g3);
    const std::array<Vec3, 4> grads{g0, g1, g2, g3};

    for(int node = 0; node < 4; ++node)
    {
        for(int col = 0; col < 3; ++col)
        {
            dfdx.block<3, 3>(3 * col, 3 * node) =
                grads[node](col) * Mat3::Identity();
        }
    }

    return dfdx;
}

void store_matrix(const Mat16& src, std::vector<double>& dst, int batch_index)
{
    const size_t offset = static_cast<size_t>(batch_index) * 16 * 16;
    for(int c = 0; c < 16; ++c)
        for(int r = 0; r < 16; ++r)
            dst[offset + static_cast<size_t>(c) * 16 + r] = src(r, c);
}
}  // namespace

ContractionCaseData make_fem12_case(const std::string& name,
                                    int                batch_count,
                                    int                seed,
                                    double             condition_scale)
{
    ContractionCaseData out;
    out.spec.name            = name;
    out.spec.op_kind         = OpKind::Fem12Assemble;
    out.spec.batch_count     = batch_count;
    out.spec.seed            = seed;
    out.spec.condition_scale = condition_scale;
    out.spec.shape           = TensorShape{12, 12, 16, 16};

    const size_t batch_stride = 16 * 16;
    out.left.resize(batch_stride * static_cast<size_t>(batch_count));
    out.middle.resize(batch_stride * static_cast<size_t>(batch_count));
    out.reference.resize(batch_stride * static_cast<size_t>(batch_count));

    std::mt19937 rng(seed);

    for(int batch = 0; batch < batch_count; ++batch)
    {
        const Vec3 x0 = random_vec(rng, 0.25);
        const Vec3 x1 = x0 + Vec3(1.0, 0.1, 0.0) + random_vec(rng, 0.05);
        const Vec3 x2 = x0 + Vec3(0.0, 1.2, 0.1) + random_vec(rng, 0.05);
        const Vec3 x3 = x0 + Vec3(0.1, 0.2, 1.1) + random_vec(rng, 0.05);

        Mat3 dm;
        dm.col(0) = x1 - x0;
        dm.col(1) = x2 - x0;
        dm.col(2) = x3 - x0;

        const Mat9x12 dfdx = tetra_dfdx(dm.inverse());
        const Mat9    k9   = random_spd9(rng, condition_scale);
        const Mat12   h12  = dfdx.transpose() * k9 * dfdx + 1.0e-4 * Mat12::Identity();

        Mat16 dpad = Mat16::Zero();
        Mat16 kpad = Mat16::Zero();
        Mat16 hpad = Mat16::Zero();

        dpad.block<9, 12>(0, 0)   = dfdx;
        kpad.block<9, 9>(0, 0)    = k9;
        hpad.block<12, 12>(0, 0)  = h12;

        store_matrix(dpad, out.left, batch);
        store_matrix(kpad, out.middle, batch);
        store_matrix(hpad, out.reference, batch);
    }

    return out;
}
}  // namespace uipc::tensor_core_lab
