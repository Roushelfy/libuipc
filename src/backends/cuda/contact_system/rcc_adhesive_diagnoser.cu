#include <contact_system/rcc_adhesive_diagnoser.h>
#include <contact_system/rcc_adhesive_coeff.h>
#include <muda/viewer/dense/dense_2d.h>
#include <contact_system/contact_models/codim_ipc_simplex_rcc_adhesive_function.h>
#include <muda/buffer/device_buffer.h>
#include <muda/launch/parallel_for.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(RCCAdhesiveDiagnoser);

// =====================================================================
// Overrider
// =====================================================================
RCCAdhesiveDiagnoserFeatureOverrider::RCCAdhesiveDiagnoserFeatureOverrider(
    RCCAdhesiveDiagnoser* diagnoser)
{
    UIPC_ASSERT(diagnoser, "RCCAdhesiveDiagnoser must not be null");
    m_diagnoser = *diagnoser;
}

void RCCAdhesiveDiagnoserFeatureOverrider::do_compute_pt_adhesion(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& triangles,
    const geometry::SimplicialComplex& prev_points,
    const geometry::SimplicialComplex& prev_triangles)
{
    m_diagnoser->compute_pt_adhesion(R, points, triangles, prev_points, prev_triangles);
}

// =====================================================================
// SimSystem
// =====================================================================
void RCCAdhesiveDiagnoser::do_build()
{
    auto overrider = std::make_shared<RCCAdhesiveDiagnoserFeatureOverrider>(this);
    auto feature   = std::make_shared<core::RCCAdhesiveDiagnoserFeature>(overrider);
    features().insert(feature);
}

namespace detail
{
    Float read_param(const geometry::Geometry& R, std::string_view name, Float default_val)
    {
        auto attr = R.instances().find<Float>(std::string{name});
        if(attr && attr->view().size() > 0)
            return attr->view()[0];
        return default_val;
    }

    template <typename T, typename D>
    void copy_back(geometry::Geometry& R,
                   std::string_view    name,
                   const D&            default_val,
                   muda::BufferView<T> buf)
    {
        std::string n{name};
        auto        attr = R.instances().find<T>(n);
        if(!attr)
            attr = R.instances().create<T>(n, T(default_val));
        buf.copy_to(view(*attr).data());
    }
}  // namespace detail

void RCCAdhesiveDiagnoser::compute_pt_adhesion(
    geometry::Geometry&                R,
    const geometry::SimplicialComplex& points,
    const geometry::SimplicialComplex& triangles,
    const geometry::SimplicialComplex& prev_points,
    const geometry::SimplicialComplex& prev_triangles)
{
    using namespace muda;

    auto h_point_pos      = points.positions().view();
    auto h_tri_pos        = triangles.positions().view();
    auto h_tri_topo       = triangles.triangles().topo().view();
    auto h_prev_point_pos = prev_points.positions().view();
    auto h_prev_tri_pos   = prev_triangles.positions().view();

    SizeT num_points    = h_point_pos.size();
    SizeT num_triangles = h_tri_topo.size();
    SizeT num_pairs     = num_points * num_triangles;

    if(num_pairs == 0)
    {
        R.instances().resize(0);
        return;
    }

    Float Cn    = detail::read_param(R, "rcc/Cn",    Float(1.0));
    Float Ct    = detail::read_param(R, "rcc/Ct",    Float(1.0));
    Float beta  = detail::read_param(R, "rcc/beta",  Float(1.0));
    Float d_hat = detail::read_param(R, "rcc/d_hat", Float(0.1));
    Float dt    = detail::read_param(R, "rcc/dt",    Float(0.01));

    DeviceBuffer<Vector3>  point_pos(num_points);
    DeviceBuffer<Vector3>  tri_pos(h_tri_pos.size());
    DeviceBuffer<Vector3i> tri_topo(num_triangles);
    DeviceBuffer<Vector3>  prev_point_pos(num_points);
    DeviceBuffer<Vector3>  prev_tri_pos(h_prev_tri_pos.size());

    point_pos.view().copy_from(h_point_pos.data());
    tri_pos.view().copy_from(h_tri_pos.data());
    tri_topo.view().copy_from(h_tri_topo.data());
    prev_point_pos.view().copy_from(h_prev_point_pos.data());
    prev_tri_pos.view().copy_from(h_prev_tri_pos.data());

    DeviceBuffer<Float>       normal_E(num_pairs);
    DeviceBuffer<Vector12>    normal_G(num_pairs);
    DeviceBuffer<Matrix12x12> normal_H(num_pairs);
    DeviceBuffer<Float>       tan_E(num_pairs);
    DeviceBuffer<Vector12>    tan_G(num_pairs);
    DeviceBuffer<Matrix12x12> tan_H(num_pairs);

    SizeT M = num_triangles;
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(
            num_pairs,
            [point_pos      = point_pos.cviewer().name("point_pos"),
             tri_pos        = tri_pos.cviewer().name("tri_pos"),
             tri_topo       = tri_topo.cviewer().name("tri_topo"),
             prev_point_pos = prev_point_pos.cviewer().name("prev_point_pos"),
             prev_tri_pos   = prev_tri_pos.cviewer().name("prev_tri_pos"),
             normal_E       = normal_E.viewer().name("normal_E"),
             normal_G       = normal_G.viewer().name("normal_G"),
             normal_H       = normal_H.viewer().name("normal_H"),
             tan_E          = tan_E.viewer().name("tan_E"),
             tan_G          = tan_G.viewer().name("tan_G"),
             tan_H          = tan_H.viewer().name("tan_H"),
             Cn, Ct, beta, d_hat, dt, M] __device__(int i) mutable
            {
                using namespace sym::codim_ipc_rcc_adhesive;

                IndexT pi = i / M;
                IndexT ti = i % M;

                const Vector3& P  = point_pos(pi);
                Vector3i       TI = tri_topo(ti);
                const Vector3& T0 = tri_pos(TI[0]);
                const Vector3& T1 = tri_pos(TI[1]);
                const Vector3& T2 = tri_pos(TI[2]);

                const Vector3& prev_P  = prev_point_pos(pi);
                const Vector3& prev_T0 = prev_tri_pos(TI[0]);
                const Vector3& prev_T1 = prev_tri_pos(TI[1]);
                const Vector3& prev_T2 = prev_tri_pos(TI[2]);

                // Normal adhesion
                normal_E(i) = PT_normal_adhesion_energy(Cn, beta, d_hat, dt, P, T0, T1, T2);

                Vector12    G_n;
                Matrix12x12 H_n;
                PT_normal_adhesion_gradient_hessian(G_n, H_n, Cn, beta, d_hat, dt, P, T0, T1, T2);
                normal_G(i) = G_n;
                normal_H(i) = H_n;

                // Tangential adhesion
                tan_E(i) = PT_tangential_adhesion_energy(
                    Ct, beta, d_hat, dt, prev_P, prev_T0, prev_T1, prev_T2, P, T0, T1, T2);

                Vector12    G_t;
                Matrix12x12 H_t;
                PT_tangential_adhesion_gradient_hessian(
                    G_t, H_t, Ct, beta, d_hat, dt, prev_P, prev_T0, prev_T1, prev_T2, P, T0, T1, T2);
                tan_G(i) = G_t;
                tan_H(i) = H_t;
            });

    R.instances().resize(num_pairs);

    detail::copy_back(R, "normal/energy", Float(0),            normal_E.view());
    detail::copy_back(R, "normal/grad",   Vector12::Zero(),    normal_G.view());
    detail::copy_back(R, "normal/hess",   Matrix12x12::Zero(), normal_H.view());
    detail::copy_back(R, "tan/energy",    Float(0),            tan_E.view());
    detail::copy_back(R, "tan/grad",      Vector12::Zero(),    tan_G.view());
    detail::copy_back(R, "tan/hess",      Matrix12x12::Zero(), tan_H.view());
}
}  // namespace uipc::backend::cuda
