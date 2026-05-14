#pragma once

#include <linear_system/socu_contact_executor.h>

#include <contact_system/contact_coeff.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <mixed_precision/cast.h>
#include <mixed_precision/policy.h>
#include <utils/codim_thickness.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance/edge_edge_mollifier.h>
#include <utils/make_spd.h>
#include <utils/primitive_d_hat.h>

namespace uipc::backend::cuda_mixed
{
template <typename StoreT>
struct SocuContactDirectSceneView
{
    muda::CDense2D<ContactCoeff> contact_tabular;
    muda::CBufferView<Vector3>   positions;
    muda::CBufferView<Vector3>   prev_positions;
    muda::CBufferView<Vector3>   rest_positions;
    muda::CBufferView<Float>     thicknesses;
    muda::CBufferView<IndexT>    contact_element_ids;
    muda::CBufferView<Float>     d_hats;
    Float                        dt = 0;
    Float                        eps_velocity = 0;

    muda::CBufferView<Vector3> half_plane_positions;
    muda::CBufferView<Vector3> half_plane_normals;
    IndexT                    half_plane_vertex_offset = 0;
};

struct SocuContactDirectSourceEntry
{
    SocuContactSourceId source_id = SocuInvalidContactSourceId;
    SocuContactModelKind model = SocuContactModelKind::SimplexNormal;
    SocuContactFamily family = SocuContactFamily::PT;
    std::uint16_t stencil_size = 0;
    std::uint32_t contact_count = 0;

    muda::CBufferView<Vector4i> stencil4;
    muda::CBufferView<Vector3i> stencil3;
    muda::CBufferView<Vector2i> stencil2;
};

struct SocuContactDirectSourceTable
{
    muda::CBufferView<SocuContactDirectSourceEntry> source_entries;

    MUDA_DEVICE SocuContactDirectSourceEntry source_for(
        const SocuContactProgramHeader& program) const noexcept
    {
        if(program.source_id == SocuInvalidContactSourceId)
            return {};
        const SizeT source_index = static_cast<SizeT>(program.source_id);
        if(source_index >= source_entries.size())
            return {};
        const auto entry = source_entries.data()[source_index];
        if(entry.source_id != program.source_id || entry.model != program.model
           || entry.family != program.family)
            return {};
        return entry;
    }
};

struct SocuContactDirectCompareProgramStats
{
    Float  max_abs_error = 0;
    Float  sum_abs_error = 0;
    IndexT mismatch_count = 0;
    IndexT compared_entry_count = 0;
};

template <typename StoreT>
struct SocuContactDirectEvaluator
{
    using Alu = ActivePolicy::AluScalar;
    using Vec3A = Eigen::Matrix<Alu, 3, 1>;
    using Vec6A = Eigen::Matrix<Alu, 6, 1>;
    using Vec9A = Eigen::Matrix<Alu, 9, 1>;
    using Vec12A = Eigen::Matrix<Alu, 12, 1>;
    using Mat3A = Eigen::Matrix<Alu, 3, 3>;
    using Mat6A = Eigen::Matrix<Alu, 6, 6>;
    using Mat9A = Eigen::Matrix<Alu, 9, 9>;
    using Mat12A = Eigen::Matrix<Alu, 12, 12>;

    SocuContactAssemblyPlanView plan;
    SocuContactDirectSceneView<StoreT> scene;
    SocuContactDirectSourceTable sources;

    MUDA_DEVICE bool supports_program(
        const SocuContactProgramHeader& program) const noexcept
    {
        const auto source = sources.source_for(program);
        return source.source_id != SocuInvalidContactSourceId
               && source.stencil_size == program.stencil_size
               && program.local_contact_id >= 0
               && static_cast<SizeT>(program.local_contact_id)
                      < static_cast<SizeT>(source.contact_count);
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> operator()(
        SizeT,
        const SocuContactProgramHeader& program) const noexcept
    {
        SocuDeterministicContactHessian<StoreT> H;
        H.stencil_size = program.stencil_size;

        const auto source = sources.source_for(program);
        if(source.source_id == SocuInvalidContactSourceId
           || program.local_contact_id < 0
           || static_cast<SizeT>(program.local_contact_id)
                  >= static_cast<SizeT>(source.contact_count))
            return H;

        switch(program.model)
        {
            case SocuContactModelKind::SimplexNormal:
                return evaluate_simplex_normal(program, source, H);
            case SocuContactModelKind::SimplexFrictional:
                return evaluate_simplex_frictional(program, source, H);
            case SocuContactModelKind::VertexHalfPlaneNormal:
                return evaluate_vertex_half_plane_normal(program, source, H);
            case SocuContactModelKind::VertexHalfPlaneFrictional:
                return evaluate_vertex_half_plane_frictional(program, source, H);
        }
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> operator()(
        const SocuContactProgramHeader& program) const noexcept
    {
        return (*this)(SizeT{0}, program);
    }

    MUDA_DEVICE IndexT local_vertex_for_global(
        const SocuContactProgramHeader& program,
        IndexT global_vertex) const noexcept
    {
        for(std::uint16_t local = 0; local < program.stencil_size; ++local)
        {
            const auto side_id = program.side_ids[local];
            if(side_id == SocuInvalidAssemblySideId
               || static_cast<SizeT>(side_id) >= plan.sides.size())
                continue;
            if(plan.sides.data()[static_cast<SizeT>(side_id)].global_vertex
               == global_vertex)
                return static_cast<IndexT>(local);
        }
        return -1;
    }

    template <typename MatT, typename StencilT>
    MUDA_DEVICE void copy_dense_matrix(
        SocuDeterministicContactHessian<StoreT>& H,
        const SocuContactProgramHeader& program,
        const StencilT& stencil,
        const MatT& dense,
        SizeT stencil_size) const noexcept
    {
        H.stencil_size = program.stencil_size;
        for(SizeT source_row = 0; source_row < stencil_size; ++source_row)
        {
            const IndexT local_row =
                local_vertex_for_global(program, stencil(source_row));
            if(local_row < 0)
                continue;
            for(SizeT source_col = 0; source_col < stencil_size; ++source_col)
            {
                const IndexT local_col =
                    local_vertex_for_global(program, stencil(source_col));
                if(local_col < 0)
                    continue;
                for(SizeT r = 0; r < 3; ++r)
                {
                    for(SizeT c = 0; c < 3; ++c)
                    {
                        const SizeT h_row =
                            static_cast<SizeT>(local_row) * 3 + r;
                        const SizeT h_col =
                            static_cast<SizeT>(local_col) * 3 + c;
                        const SizeT dense_row = source_row * 3 + r;
                        const SizeT dense_col = source_col * 3 + c;
                        H.values[h_row
                                 * SocuDeterministicContactHessian<
                                       StoreT>::MaxDofCount
                                 + h_col] =
                            safe_cast<StoreT>(dense(dense_row, dense_col));
                    }
                }
            }
        }
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT> evaluate_simplex_normal(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        using namespace sym::codim_ipc_simplex_contact;

        const SizeT contact_id = static_cast<SizeT>(program.local_contact_id);
        switch(program.family)
        {
            case SocuContactFamily::PT:
            {
                const auto PT = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[PT[0]],
                                 scene.contact_element_ids.data()[PT[1]],
                                 scene.contact_element_ids.data()[PT[2]],
                                 scene.contact_element_ids.data()[PT[3]]};
                const Alu kt2 =
                    safe_cast<Alu>(PT_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Vec3A P =
                    scene.positions.data()[PT[0]].template cast<Alu>();
                const Vec3A T0 =
                    scene.positions.data()[PT[1]].template cast<Alu>();
                const Vec3A T1 =
                    scene.positions.data()[PT[2]].template cast<Alu>();
                const Vec3A T2 =
                    scene.positions.data()[PT[3]].template cast<Alu>();
                const Alu thickness = safe_cast<Alu>(PT_thickness(
                    scene.thicknesses.data()[PT[0]],
                    scene.thicknesses.data()[PT[1]],
                    scene.thicknesses.data()[PT[2]],
                    scene.thicknesses.data()[PT[3]]));
                const Alu d_hat = safe_cast<Alu>(PT_d_hat(
                    scene.d_hats.data()[PT[0]],
                    scene.d_hats.data()[PT[1]],
                    scene.d_hats.data()[PT[2]],
                    scene.d_hats.data()[PT[3]]));
                const Vector4i flag = distance::point_triangle_distance_flag(
                    scene.positions.data()[PT[0]],
                    scene.positions.data()[PT[1]],
                    scene.positions.data()[PT[2]],
                    scene.positions.data()[PT[3]]);
                Vec12A G;
                Mat12A dense;
                PT_barrier_gradient_hessian(
                    G, dense, flag, kt2, d_hat, thickness, P, T0, T1, T2);
                make_spd(dense);
                copy_dense_matrix(H, program, PT, dense, 4);
                return H;
            }
            case SocuContactFamily::EE:
            {
                const auto EE = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[EE[0]],
                                 scene.contact_element_ids.data()[EE[1]],
                                 scene.contact_element_ids.data()[EE[2]],
                                 scene.contact_element_ids.data()[EE[3]]};
                const Alu kt2 =
                    safe_cast<Alu>(EE_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Vec3A E0 =
                    scene.positions.data()[EE[0]].template cast<Alu>();
                const Vec3A E1 =
                    scene.positions.data()[EE[1]].template cast<Alu>();
                const Vec3A E2 =
                    scene.positions.data()[EE[2]].template cast<Alu>();
                const Vec3A E3 =
                    scene.positions.data()[EE[3]].template cast<Alu>();
                const Vec3A R0 =
                    scene.rest_positions.data()[EE[0]].template cast<Alu>();
                const Vec3A R1 =
                    scene.rest_positions.data()[EE[1]].template cast<Alu>();
                const Vec3A R2 =
                    scene.rest_positions.data()[EE[2]].template cast<Alu>();
                const Vec3A R3 =
                    scene.rest_positions.data()[EE[3]].template cast<Alu>();
                const Alu thickness = safe_cast<Alu>(EE_thickness(
                    scene.thicknesses.data()[EE[0]],
                    scene.thicknesses.data()[EE[1]],
                    scene.thicknesses.data()[EE[2]],
                    scene.thicknesses.data()[EE[3]]));
                const Alu d_hat = safe_cast<Alu>(EE_d_hat(
                    scene.d_hats.data()[EE[0]],
                    scene.d_hats.data()[EE[1]],
                    scene.d_hats.data()[EE[2]],
                    scene.d_hats.data()[EE[3]]));
                const Vector4i flag = distance::edge_edge_distance_flag(
                    scene.positions.data()[EE[0]],
                    scene.positions.data()[EE[1]],
                    scene.positions.data()[EE[2]],
                    scene.positions.data()[EE[3]]);
                Vec12A G;
                Mat12A dense;
                mollified_EE_barrier_gradient_hessian(G,
                                                       dense,
                                                       flag,
                                                       kt2,
                                                       d_hat,
                                                       thickness,
                                                       R0,
                                                       R1,
                                                       R2,
                                                       R3,
                                                       E0,
                                                       E1,
                                                       E2,
                                                       E3);
                make_spd(dense);
                copy_dense_matrix(H, program, EE, dense, 4);
                return H;
            }
            case SocuContactFamily::PE:
            {
                const auto PE = source.stencil3.data()[contact_id];
                Vector3i cids = {scene.contact_element_ids.data()[PE[0]],
                                 scene.contact_element_ids.data()[PE[1]],
                                 scene.contact_element_ids.data()[PE[2]]};
                const Alu kt2 =
                    safe_cast<Alu>(PE_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Vec3A P =
                    scene.positions.data()[PE[0]].template cast<Alu>();
                const Vec3A E0 =
                    scene.positions.data()[PE[1]].template cast<Alu>();
                const Vec3A E1 =
                    scene.positions.data()[PE[2]].template cast<Alu>();
                const Alu thickness = safe_cast<Alu>(PE_thickness(
                    scene.thicknesses.data()[PE[0]],
                    scene.thicknesses.data()[PE[1]],
                    scene.thicknesses.data()[PE[2]]));
                const Alu d_hat = safe_cast<Alu>(PE_d_hat(
                    scene.d_hats.data()[PE[0]],
                    scene.d_hats.data()[PE[1]],
                    scene.d_hats.data()[PE[2]]));
                const Vector3i flag = distance::point_edge_distance_flag(
                    scene.positions.data()[PE[0]],
                    scene.positions.data()[PE[1]],
                    scene.positions.data()[PE[2]]);
                Vec9A G;
                Mat9A dense;
                PE_barrier_gradient_hessian(
                    G, dense, flag, kt2, d_hat, thickness, P, E0, E1);
                make_spd(dense);
                copy_dense_matrix(H, program, PE, dense, 3);
                return H;
            }
            case SocuContactFamily::PP:
            {
                const auto PP = source.stencil2.data()[contact_id];
                Vector2i cids = {scene.contact_element_ids.data()[PP[0]],
                                 scene.contact_element_ids.data()[PP[1]]};
                const Alu kt2 =
                    safe_cast<Alu>(PP_kappa(scene.contact_tabular, cids)
                                   * scene.dt * scene.dt);
                const Vec3A P0 =
                    scene.positions.data()[PP[0]].template cast<Alu>();
                const Vec3A P1 =
                    scene.positions.data()[PP[1]].template cast<Alu>();
                const Alu thickness = safe_cast<Alu>(
                    PP_thickness(scene.thicknesses.data()[PP[0]],
                                 scene.thicknesses.data()[PP[1]]));
                const Alu d_hat = safe_cast<Alu>(
                    PP_d_hat(scene.d_hats.data()[PP[0]],
                             scene.d_hats.data()[PP[1]]));
                const Vector2i flag = distance::point_point_distance_flag(
                    scene.positions.data()[PP[0]],
                    scene.positions.data()[PP[1]]);
                Vec6A G;
                Mat6A dense;
                PP_barrier_gradient_hessian(
                    G, dense, flag, kt2, d_hat, thickness, P0, P1);
                make_spd(dense);
                copy_dense_matrix(H, program, PP, dense, 2);
                return H;
            }
            case SocuContactFamily::PH:
                return H;
        }
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT>
    evaluate_simplex_frictional(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        using namespace sym::codim_ipc_contact;

        const SizeT contact_id = static_cast<SizeT>(program.local_contact_id);
        const Alu epsvdt = safe_cast<Alu>(scene.eps_velocity * scene.dt);
        switch(program.family)
        {
            case SocuContactFamily::PT:
            {
                const auto PT = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[PT[0]],
                                 scene.contact_element_ids.data()[PT[1]],
                                 scene.contact_element_ids.data()[PT[2]],
                                 scene.contact_element_ids.data()[PT[3]]};
                const auto coeff = PT_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu mu = safe_cast<Alu>(coeff.mu);
                const Alu thickness = safe_cast<Alu>(PT_thickness(
                    scene.thicknesses.data()[PT[0]],
                    scene.thicknesses.data()[PT[1]],
                    scene.thicknesses.data()[PT[2]],
                    scene.thicknesses.data()[PT[3]]));
                const Alu d_hat = safe_cast<Alu>(PT_d_hat(
                    scene.d_hats.data()[PT[0]],
                    scene.d_hats.data()[PT[1]],
                    scene.d_hats.data()[PT[2]],
                    scene.d_hats.data()[PT[3]]));
                Vec12A G;
                Mat12A dense;
                PT_friction_gradient_hessian(
                    G,
                    dense,
                    kt2,
                    d_hat,
                    thickness,
                    mu,
                    epsvdt,
                    scene.prev_positions.data()[PT[0]].template cast<Alu>(),
                    scene.prev_positions.data()[PT[1]].template cast<Alu>(),
                    scene.prev_positions.data()[PT[2]].template cast<Alu>(),
                    scene.prev_positions.data()[PT[3]].template cast<Alu>(),
                    scene.positions.data()[PT[0]].template cast<Alu>(),
                    scene.positions.data()[PT[1]].template cast<Alu>(),
                    scene.positions.data()[PT[2]].template cast<Alu>(),
                    scene.positions.data()[PT[3]].template cast<Alu>());
                make_spd(dense);
                copy_dense_matrix(H, program, PT, dense, 4);
                return H;
            }
            case SocuContactFamily::EE:
            {
                const auto EE = source.stencil4.data()[contact_id];
                Vector4i cids = {scene.contact_element_ids.data()[EE[0]],
                                 scene.contact_element_ids.data()[EE[1]],
                                 scene.contact_element_ids.data()[EE[2]],
                                 scene.contact_element_ids.data()[EE[3]]};
                const auto coeff = EE_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu mu = safe_cast<Alu>(coeff.mu);
                const Alu thickness = safe_cast<Alu>(EE_thickness(
                    scene.thicknesses.data()[EE[0]],
                    scene.thicknesses.data()[EE[1]],
                    scene.thicknesses.data()[EE[2]],
                    scene.thicknesses.data()[EE[3]]));
                const Alu d_hat = safe_cast<Alu>(EE_d_hat(
                    scene.d_hats.data()[EE[0]],
                    scene.d_hats.data()[EE[1]],
                    scene.d_hats.data()[EE[2]],
                    scene.d_hats.data()[EE[3]]));

                const Vec3A rest0 =
                    scene.rest_positions.data()[EE[0]].template cast<Alu>();
                const Vec3A rest1 =
                    scene.rest_positions.data()[EE[1]].template cast<Alu>();
                const Vec3A rest2 =
                    scene.rest_positions.data()[EE[2]].template cast<Alu>();
                const Vec3A rest3 =
                    scene.rest_positions.data()[EE[3]].template cast<Alu>();
                Alu eps_x;
                distance::edge_edge_mollifier_threshold(
                    rest0, rest1, rest2, rest3, static_cast<Alu>(1e-3), eps_x);

                const Vec3A prev0 =
                    scene.prev_positions.data()[EE[0]].template cast<Alu>();
                const Vec3A prev1 =
                    scene.prev_positions.data()[EE[1]].template cast<Alu>();
                const Vec3A prev2 =
                    scene.prev_positions.data()[EE[2]].template cast<Alu>();
                const Vec3A prev3 =
                    scene.prev_positions.data()[EE[3]].template cast<Alu>();
                Vec12A G;
                Mat12A dense;
                const bool mollified =
                    distance::need_mollify(prev0, prev1, prev2, prev3, eps_x);
                if(mollified)
                {
                    G.setZero();
                    dense.setZero();
                }
                else
                {
                    EE_friction_gradient_hessian(
                        G,
                        dense,
                        kt2,
                        d_hat,
                        thickness,
                        mu,
                        epsvdt,
                        prev0,
                        prev1,
                        prev2,
                        prev3,
                        scene.positions.data()[EE[0]].template cast<Alu>(),
                        scene.positions.data()[EE[1]].template cast<Alu>(),
                        scene.positions.data()[EE[2]].template cast<Alu>(),
                        scene.positions.data()[EE[3]].template cast<Alu>());
                    make_spd(dense);
                }
                copy_dense_matrix(H, program, EE, dense, 4);
                return H;
            }
            case SocuContactFamily::PE:
            {
                const auto PE = source.stencil3.data()[contact_id];
                Vector3i cids = {scene.contact_element_ids.data()[PE[0]],
                                 scene.contact_element_ids.data()[PE[1]],
                                 scene.contact_element_ids.data()[PE[2]]};
                const auto coeff = PE_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu mu = safe_cast<Alu>(coeff.mu);
                const Alu thickness = safe_cast<Alu>(PE_thickness(
                    scene.thicknesses.data()[PE[0]],
                    scene.thicknesses.data()[PE[1]],
                    scene.thicknesses.data()[PE[2]]));
                const Alu d_hat = safe_cast<Alu>(PE_d_hat(
                    scene.d_hats.data()[PE[0]],
                    scene.d_hats.data()[PE[1]],
                    scene.d_hats.data()[PE[2]]));
                Vec9A G;
                Mat9A dense;
                PE_friction_gradient_hessian(
                    G,
                    dense,
                    kt2,
                    d_hat,
                    thickness,
                    mu,
                    epsvdt,
                    scene.prev_positions.data()[PE[0]].template cast<Alu>(),
                    scene.prev_positions.data()[PE[1]].template cast<Alu>(),
                    scene.prev_positions.data()[PE[2]].template cast<Alu>(),
                    scene.positions.data()[PE[0]].template cast<Alu>(),
                    scene.positions.data()[PE[1]].template cast<Alu>(),
                    scene.positions.data()[PE[2]].template cast<Alu>());
                make_spd(dense);
                copy_dense_matrix(H, program, PE, dense, 3);
                return H;
            }
            case SocuContactFamily::PP:
            {
                const auto PP = source.stencil2.data()[contact_id];
                Vector2i cids = {scene.contact_element_ids.data()[PP[0]],
                                 scene.contact_element_ids.data()[PP[1]]};
                const auto coeff = PP_contact_coeff(scene.contact_tabular, cids);
                const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
                const Alu mu = safe_cast<Alu>(coeff.mu);
                const Alu thickness = safe_cast<Alu>(
                    PP_thickness(scene.thicknesses.data()[PP[0]],
                                 scene.thicknesses.data()[PP[1]]));
                const Alu d_hat = safe_cast<Alu>(
                    PP_d_hat(scene.d_hats.data()[PP[0]],
                             scene.d_hats.data()[PP[1]]));
                Vec6A G;
                Mat6A dense;
                PP_friction_gradient_hessian(
                    G,
                    dense,
                    kt2,
                    d_hat,
                    thickness,
                    mu,
                    epsvdt,
                    scene.prev_positions.data()[PP[0]].template cast<Alu>(),
                    scene.prev_positions.data()[PP[1]].template cast<Alu>(),
                    scene.positions.data()[PP[0]].template cast<Alu>(),
                    scene.positions.data()[PP[1]].template cast<Alu>());
                make_spd(dense);
                copy_dense_matrix(H, program, PP, dense, 2);
                return H;
            }
            case SocuContactFamily::PH:
                return H;
        }
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT>
    evaluate_vertex_half_plane_normal(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        if(program.family != SocuContactFamily::PH)
            return H;
        using namespace sym::ipc_vertex_half_contact;

        const auto PH =
            source.stencil2.data()[static_cast<SizeT>(program.local_contact_id)];
        const IndexT vI = PH(0);
        const IndexT HI = PH(1);
        const ContactCoeff coeff =
            scene.contact_tabular(scene.contact_element_ids.data()[vI],
                                  scene.contact_element_ids.data()[HI
                                      + scene.half_plane_vertex_offset]);
        const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
        const Alu thickness =
            safe_cast<Alu>(scene.thicknesses.data()[vI]);
        const Alu d_hat = safe_cast<Alu>(scene.d_hats.data()[vI]);
        Vec3A G;
        Mat3A dense;
        PH_barrier_gradient_hessian(
            G,
            dense,
            kt2,
            d_hat,
            thickness,
            scene.positions.data()[vI].template cast<Alu>(),
            scene.half_plane_positions.data()[HI].template cast<Alu>(),
            scene.half_plane_normals.data()[HI].template cast<Alu>());
        copy_dense_matrix(H, program, PH, dense, 1);
        return H;
    }

    MUDA_DEVICE SocuDeterministicContactHessian<StoreT>
    evaluate_vertex_half_plane_frictional(
        const SocuContactProgramHeader& program,
        const SocuContactDirectSourceEntry& source,
        SocuDeterministicContactHessian<StoreT> H) const noexcept
    {
        if(program.family != SocuContactFamily::PH)
            return H;
        using namespace sym::ipc_vertex_half_contact;

        const auto PH =
            source.stencil2.data()[static_cast<SizeT>(program.local_contact_id)];
        const IndexT vI = PH(0);
        const IndexT HI = PH(1);
        const ContactCoeff coeff =
            scene.contact_tabular(scene.contact_element_ids.data()[vI],
                                  scene.contact_element_ids.data()[HI
                                      + scene.half_plane_vertex_offset]);
        const Alu kt2 = safe_cast<Alu>(coeff.kappa * scene.dt * scene.dt);
        const Alu mu = safe_cast<Alu>(coeff.mu);
        const Alu thickness =
            safe_cast<Alu>(scene.thicknesses.data()[vI]);
        const Alu d_hat = safe_cast<Alu>(scene.d_hats.data()[vI]);
        Vec3A G;
        Mat3A dense;
        PH_friction_gradient_hessian(
            G,
            dense,
            kt2,
            d_hat,
            thickness,
            mu,
            safe_cast<Alu>(scene.eps_velocity * scene.dt),
            scene.prev_positions.data()[vI].template cast<Alu>(),
            scene.positions.data()[vI].template cast<Alu>(),
            scene.half_plane_positions.data()[HI].template cast<Alu>(),
            scene.half_plane_normals.data()[HI].template cast<Alu>());
        make_spd(dense);
        copy_dense_matrix(H, program, PH, dense, 1);
        return H;
    }
};

template <typename StoreT, typename EvaluatorT>
__global__ void socu_contact_direct_evaluate_programs_kernel(
    SocuContactAssemblyPlanView plan,
    EvaluatorT evaluator,
    muda::BufferView<SocuDeterministicContactHessian<StoreT>> hessians,
    muda::BufferView<IndexT> unsupported_program_flags)
{
    const SizeT program_id =
        static_cast<SizeT>(blockIdx.x) * static_cast<SizeT>(blockDim.x)
        + static_cast<SizeT>(threadIdx.x);
    if(program_id >= plan.programs.size() || program_id >= hessians.size())
        return;
    const auto program = plan.programs.data()[program_id];
    const bool supported = evaluator.supports_program(program);
    if(unsupported_program_flags.data() != nullptr
       && program_id < unsupported_program_flags.size())
        unsupported_program_flags.data()[program_id] =
            supported ? IndexT{0} : IndexT{1};
    hessians.data()[program_id] = evaluator(program_id, program);
}

template <typename StoreT, typename EvaluatorT>
void launch_socu_contact_direct_evaluate_programs(
    SocuContactAssemblyPlanView plan,
    EvaluatorT evaluator,
    muda::BufferView<SocuDeterministicContactHessian<StoreT>> hessians,
    muda::BufferView<IndexT> unsupported_program_flags = {},
    cudaStream_t stream = cudaStreamLegacy)
{
    if(plan.programs.size() == 0 || hessians.size() == 0)
        return;

    constexpr unsigned int block_dim = 128;
    const SizeT count =
        plan.programs.size() < hessians.size() ? plan.programs.size()
                                               : hessians.size();
    const unsigned int grid_dim =
        static_cast<unsigned int>((count + block_dim - 1) / block_dim);
    socu_contact_direct_evaluate_programs_kernel<StoreT, EvaluatorT>
        <<<grid_dim, block_dim, 0, socu_contact_executor_stream(stream)>>>(
            plan,
            evaluator,
            hessians,
            unsupported_program_flags);
}

template <typename StoreT, typename ReferenceEvaluatorT>
__global__ void socu_contact_replace_direct_unsupported_programs_kernel(
    SocuContactAssemblyPlanView plan,
    muda::BufferView<SocuDeterministicContactHessian<StoreT>> hessians,
    muda::CBufferView<IndexT> unsupported_program_flags,
    ReferenceEvaluatorT reference_evaluator)
{
    const SizeT program_id =
        static_cast<SizeT>(blockIdx.x) * static_cast<SizeT>(blockDim.x)
        + static_cast<SizeT>(threadIdx.x);
    if(program_id >= plan.programs.size() || program_id >= hessians.size()
       || program_id >= unsupported_program_flags.size())
        return;
    if(unsupported_program_flags.data()[program_id] == IndexT{0})
        return;

    const auto program = plan.programs.data()[program_id];
    hessians.data()[program_id] = reference_evaluator(program_id, program);
}

template <typename StoreT, typename ReferenceEvaluatorT>
void launch_socu_contact_replace_direct_unsupported_programs(
    SocuContactAssemblyPlanView plan,
    muda::BufferView<SocuDeterministicContactHessian<StoreT>> hessians,
    muda::CBufferView<IndexT> unsupported_program_flags,
    ReferenceEvaluatorT reference_evaluator,
    cudaStream_t stream = cudaStreamLegacy)
{
    if(plan.programs.size() == 0 || hessians.size() == 0
       || unsupported_program_flags.size() == 0)
        return;

    constexpr unsigned int block_dim = 128;
    const SizeT count =
        plan.programs.size() < hessians.size()
            ? (plan.programs.size() < unsupported_program_flags.size()
                   ? plan.programs.size()
                   : unsupported_program_flags.size())
            : (hessians.size() < unsupported_program_flags.size()
                   ? hessians.size()
                   : unsupported_program_flags.size());
    const unsigned int grid_dim =
        static_cast<unsigned int>((count + block_dim - 1) / block_dim);
    socu_contact_replace_direct_unsupported_programs_kernel<
        StoreT,
        ReferenceEvaluatorT>
        <<<grid_dim, block_dim, 0, socu_contact_executor_stream(stream)>>>(
            plan,
            hessians,
            unsupported_program_flags,
            reference_evaluator);
}

template <typename StoreT, typename ReferenceEvaluatorT>
__global__ void socu_contact_compare_direct_triplet_programs_kernel(
    SocuContactAssemblyPlanView plan,
    muda::CBufferView<SocuDeterministicContactHessian<StoreT>> direct_hessians,
    ReferenceEvaluatorT reference_evaluator,
    muda::BufferView<SocuContactDirectCompareProgramStats> stats,
    Float tolerance)
{
    const SizeT program_id =
        static_cast<SizeT>(blockIdx.x) * static_cast<SizeT>(blockDim.x)
        + static_cast<SizeT>(threadIdx.x);
    if(program_id >= plan.programs.size() || program_id >= direct_hessians.size()
       || program_id >= stats.size())
        return;

    const auto program = plan.programs.data()[program_id];
    const auto direct = direct_hessians.data()[program_id];
    const auto reference = reference_evaluator(program_id, program);

    const SizeT dof_count =
        static_cast<SizeT>(program.stencil_size) * SizeT{3};
    Float  max_abs_error = 0;
    Float  sum_abs_error = 0;
    IndexT mismatch_count = 0;
    IndexT compared_entry_count = 0;

    for(SizeT row = 0;
        row < dof_count
        && row < SocuDeterministicContactHessian<StoreT>::MaxDofCount;
        ++row)
    {
        for(SizeT col = 0;
            col < dof_count
            && col < SocuDeterministicContactHessian<StoreT>::MaxDofCount;
            ++col)
        {
            const Float diff = static_cast<Float>(direct(row, col))
                               - static_cast<Float>(reference(row, col));
            const Float abs_error = diff >= Float{0} ? diff : -diff;
            if(abs_error > max_abs_error)
                max_abs_error = abs_error;
            sum_abs_error += abs_error;
            if(abs_error > tolerance)
                ++mismatch_count;
            ++compared_entry_count;
        }
    }

    stats.data()[program_id] = SocuContactDirectCompareProgramStats{
        max_abs_error,
        sum_abs_error,
        mismatch_count,
        compared_entry_count};
}

template <typename StoreT, typename ReferenceEvaluatorT>
void launch_socu_contact_compare_direct_triplet_programs(
    SocuContactAssemblyPlanView plan,
    muda::CBufferView<SocuDeterministicContactHessian<StoreT>> direct_hessians,
    ReferenceEvaluatorT reference_evaluator,
    muda::BufferView<SocuContactDirectCompareProgramStats> stats,
    Float tolerance,
    cudaStream_t stream = cudaStreamLegacy)
{
    if(plan.programs.size() == 0 || direct_hessians.size() == 0
       || stats.size() == 0)
        return;

    constexpr unsigned int block_dim = 128;
    const SizeT count =
        plan.programs.size() < direct_hessians.size()
            ? (plan.programs.size() < stats.size() ? plan.programs.size()
                                                   : stats.size())
            : (direct_hessians.size() < stats.size() ? direct_hessians.size()
                                                     : stats.size());
    const unsigned int grid_dim =
        static_cast<unsigned int>((count + block_dim - 1) / block_dim);
    socu_contact_compare_direct_triplet_programs_kernel<
        StoreT,
        ReferenceEvaluatorT>
        <<<grid_dim, block_dim, 0, socu_contact_executor_stream(stream)>>>(
            plan,
            direct_hessians,
            reference_evaluator,
            stats,
            tolerance);
}
}  // namespace uipc::backend::cuda_mixed
