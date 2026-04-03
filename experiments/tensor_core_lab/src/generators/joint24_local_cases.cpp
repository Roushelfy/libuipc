#include <tcl/case_spec.h>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include <cmath>
#include <random>

namespace uipc::tensor_core_lab
{
namespace
{
using Vec3    = Eigen::Vector3d;
using Mat3    = Eigen::Matrix3d;
using Mat9    = Eigen::Matrix<double, 9, 9>;
using Mat24   = Eigen::Matrix<double, 24, 24>;
using Mat3x12 = Eigen::Matrix<double, 3, 12>;
using Mat3x24 = Eigen::Matrix<double, 3, 24>;
using Mat9x24 = Eigen::Matrix<double, 9, 24>;
using Mat32   = Eigen::Matrix<double, 32, 32>;

Vec3 random_vec(std::mt19937& rng, double scale)
{
    std::uniform_real_distribution<double> dist(-scale, scale);
    return {dist(rng), dist(rng), dist(rng)};
}

template <int N>
Eigen::Matrix<double, N, N> random_spd(std::mt19937& rng, double condition_scale)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::Matrix<double, N, N> raw;
    for(int c = 0; c < N; ++c)
        for(int r = 0; r < N; ++r)
            raw(r, c) = dist(rng);

    Eigen::HouseholderQR<Eigen::Matrix<double, N, N>> qr(raw);
    const auto q = qr.householderQ() * Eigen::Matrix<double, N, N>::Identity();

    Eigen::Matrix<double, N, 1> diag;
    for(int i = 0; i < N; ++i)
    {
        const double t = (N == 1) ? 0.0 : static_cast<double>(i) / (N - 1);
        diag(i)        = std::pow(condition_scale, t);
    }

    return q * diag.asDiagonal() * q.transpose()
           + 1.0e-2 * Eigen::Matrix<double, N, N>::Identity();
}

Mat3x12 make_abd_jacobian(const Vec3& x)
{
    Mat3x12 ret = Mat3x12::Zero();
    ret(0, 0) = 1.0;
    ret(1, 1) = 1.0;
    ret(2, 2) = 1.0;
    ret.block<1, 3>(0, 3) = x.transpose();
    ret.block<1, 3>(1, 6) = x.transpose();
    ret.block<1, 3>(2, 9) = x.transpose();
    return ret;
}

void store_matrix(const Mat32& src, std::vector<double>& dst, int batch_index)
{
    const size_t offset = static_cast<size_t>(batch_index) * 32 * 32;
    for(int c = 0; c < 32; ++c)
        for(int r = 0; r < 32; ++r)
            dst[offset + static_cast<size_t>(c) * 32 + r] = src(r, c);
}
}  // namespace

ContractionCaseData make_joint24_case(const std::string& name,
                                      int                batch_count,
                                      int                seed,
                                      double             condition_scale)
{
    ContractionCaseData out;
    out.spec.name            = name;
    out.spec.op_kind         = OpKind::Joint24Assemble;
    out.spec.batch_count     = batch_count;
    out.spec.seed            = seed;
    out.spec.condition_scale = condition_scale;
    out.spec.shape           = TensorShape{24, 24, 32, 32};

    const size_t batch_stride = 32 * 32;
    out.left.resize(batch_stride * static_cast<size_t>(batch_count));
    out.middle.resize(batch_stride * static_cast<size_t>(batch_count));
    out.right.resize(batch_stride * static_cast<size_t>(batch_count));
    out.aux.resize(batch_stride * static_cast<size_t>(batch_count));
    out.reference.resize(batch_stride * static_cast<size_t>(batch_count));

    std::mt19937 rng(seed);

    for(int batch = 0; batch < batch_count; ++batch)
    {
        Mat9x24 jr = Mat9x24::Zero();
        for(int i = 0; i < 3; ++i)
        {
            const Mat3x12 ji = make_abd_jacobian(random_vec(rng, 0.8));
            const Mat3x12 jj = make_abd_jacobian(random_vec(rng, 0.8));
            jr.block<3, 12>(3 * i, 0)  = ji;
            jr.block<3, 12>(3 * i, 12) = -jj;
        }

        Mat3x24 jt = Mat3x24::Zero();
        jt.block<3, 12>(0, 0)  = make_abd_jacobian(random_vec(rng, 0.5));
        jt.block<3, 12>(0, 12) = -make_abd_jacobian(random_vec(rng, 0.5));

        const Mat9  hr  = random_spd<9>(rng, condition_scale);
        const Mat3  ht  = random_spd<3>(rng, condition_scale);
        const Mat24 h24 = jr.transpose() * hr * jr + jt.transpose() * ht * jt
                          + 1.0e-4 * Mat24::Identity();

        Mat32 jr_pad = Mat32::Zero();
        Mat32 hr_pad = Mat32::Zero();
        Mat32 jt_pad = Mat32::Zero();
        Mat32 ht_pad = Mat32::Zero();
        Mat32 h_pad  = Mat32::Zero();

        jr_pad.block<9, 24>(0, 0)   = jr;
        hr_pad.block<9, 9>(0, 0)    = hr;
        jt_pad.block<3, 24>(0, 0)   = jt;
        ht_pad.block<3, 3>(0, 0)    = ht;
        h_pad.block<24, 24>(0, 0)   = h24;

        store_matrix(jr_pad, out.left, batch);
        store_matrix(hr_pad, out.middle, batch);
        store_matrix(jt_pad, out.right, batch);
        store_matrix(ht_pad, out.aux, batch);
        store_matrix(h_pad, out.reference, batch);
    }

    return out;
}
}  // namespace uipc::tensor_core_lab
