#pragma once
#include <collision_detection/trajectory_filter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_body_manager.h>
#include <contact_system/global_contact_manager.h>
#include <muda/buffer/device_buffer.h>
#include <utils/dump_utils.h>

namespace uipc::backend::cuda
{
// One active vertex-triangle (VT) primitive: the full (point, t0, t1, t2)
// topology plus its closest-feature classification flag (the Vector4i from
// point_triangle_distance_flag; degenerate_point_triangle(flag) -> 2=PP /
// 3=PE / 4=PT-interior and the sub-feature offsets). Emitted ADDITIVELY by
// each simplex filter's filter_active for every active VT candidate (before
// the dim-switch reduces it), so RCC adhesion can run per-VT-primitive while
// the barrier/friction reduced lists stay untouched. The lagged flag is
// snapshotted with the topology so the feature classification is fixed across
// the step (consistent with friction's lagged basis). Trivially copyable so
// it can ride a DeviceSelect compaction.
struct ActiveVT
{
    Vector4i topo;
    Vector4i flag;
};

class SimplexTrajectoryFilter : public TrajectoryFilter
{
  public:
    using TrajectoryFilter::TrajectoryFilter;

    class Impl;

    class BuildInfo
    {
      public:
    };

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl) noexcept
            : m_impl(impl)
        {
        }

        Float d_hat() const noexcept;

        // Vertex Attributes

        /**
         * @brief Vertex Id to Body Id mapping.
         */
        muda::CBufferView<Float>    d_hats() const noexcept;
        muda::CBufferView<IndexT>   v2b() const noexcept;
        muda::CBufferView<Vector3>  positions() const noexcept;
        muda::CBufferView<Vector3>  rest_positions() const noexcept;
        muda::CBufferView<Float>    thicknesses() const noexcept;
        muda::CBufferView<IndexT>   dimensions() const noexcept;
        muda::CBufferView<IndexT>   contact_element_ids() const noexcept;
        muda::CBufferView<IndexT>   subscene_element_ids() const noexcept;
        muda::CBuffer2DView<IndexT> contact_mask_tabular() const noexcept;
        muda::CBuffer2DView<IndexT> subscene_mask_tabular() const noexcept;
        // Body Attributes

        /**
         * @brief Tell if the body needs self-collision
         */
        muda::CBufferView<IndexT> body_self_collision() const noexcept;

        // Topologies

        muda::CBufferView<IndexT>   codim_vertices() const noexcept;
        muda::CBufferView<IndexT>   surf_vertices() const noexcept;
        muda::CBufferView<Vector2i> surf_edges() const noexcept;
        muda::CBufferView<Vector3i> surf_triangles() const noexcept;

      protected:
        friend class SimplexTrajectoryFilter;
        Impl* m_impl = nullptr;
    };

    class DetectInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        Float alpha() const noexcept { return m_alpha; }

        muda::CBufferView<Vector3> displacements() const noexcept;

        // RCC bonded-PT pre-CCD filter inputs (see rcc_bonded_pt_system).
        muda::CBufferView<U64> rcc_bonded_pt_locked_keys() const noexcept;
        bool                   rcc_bonded_pt_skip_ccd() const noexcept;

      private:
        friend class SimplexTrajectoryFilter;
        Float m_alpha = 0.0;
    };

    class FilterActiveInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        /**
         * @brief Candidate point-triangle pairs.
         */
        void PTs(muda::CBufferView<Vector4i> PTs) noexcept;
        /**
         * @brief Candidate edge-edge pairs.
         */
        void EEs(muda::CBufferView<Vector4i> EEs) noexcept;
        /**
         * @brief Candidate point-edge pairs.
         */
        void PEs(muda::CBufferView<Vector3i> PEs) noexcept;
        /**
         * @brief Candidate point-point pairs.
         */
        void PPs(muda::CBufferView<Vector2i> PPs) noexcept;
        /**
         * @brief Active vertex-triangle primitives (full topo + feature flag),
         * additive and parallel to PTs/PEs/PPs. One entry per active VT
         * candidate regardless of closest-feature reduction.
         */
        void VTs(muda::CBufferView<ActiveVT> VTs) noexcept;
    };

    class FilterTOIInfo : public DetectInfo
    {
      public:
        using DetectInfo::DetectInfo;

        muda::VarView<Float> toi() noexcept;

      private:
        friend class SimplexTrajectoryFilter;
        muda::VarView<Float> m_toi;
    };

    class Impl
    {
      public:
        void record_friction_candidates(GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info);
        void label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info);
        void filter_rcc_bonded_pt_locked_active_pairs();
        void set_rcc_bonded_pt_locked_keys(muda::CBufferView<U64> locked_keys) noexcept;
        void clear_rcc_bonded_pt_locked_keys() noexcept;
        SizeT rcc_bonded_pt_filter_skipped_count() const noexcept;
        SizeT rcc_bonded_pt_filter_generation() const noexcept;
        bool dump(DumpInfo& info);
        bool try_recover(RecoverInfo& info);
        void apply_recover(RecoverInfo& info);
        void clear_recover(RecoverInfo& info);

        SimSystemSlot<GlobalVertexManager> global_vertex_manager;
        SimSystemSlot<GlobalSimplicialSurfaceManager> global_simplicial_surface_manager;
        SimSystemSlot<GlobalContactManager> global_contact_manager;
        SimSystemSlot<GlobalBodyManager>    global_body_manager;

        muda::CBufferView<Vector4i> PTs;
        muda::CBufferView<Vector4i> EEs;
        muda::CBufferView<Vector3i> PEs;
        muda::CBufferView<Vector2i> PPs;
        muda::CBufferView<ActiveVT>  VTs;

        muda::DeviceBuffer<Vector4i> friction_PT;
        muda::DeviceBuffer<Vector4i> friction_EE;
        muda::DeviceBuffer<Vector3i> friction_PE;
        muda::DeviceBuffer<Vector2i> friction_PP;
        muda::DeviceBuffer<ActiveVT> friction_VT;

        muda::DeviceBuffer<Vector4i> recovered_PT;
        muda::DeviceBuffer<Vector4i> recovered_EE;
        muda::DeviceBuffer<Vector3i> recovered_PE;
        muda::DeviceBuffer<Vector2i> recovered_PP;

        muda::CBufferView<U64>       rcc_bonded_pt_locked_keys;
        muda::DeviceBuffer<Vector4i> rcc_bonded_pt_unlocked_PT;
        muda::DeviceVar<IndexT>      rcc_bonded_pt_unlocked_PT_count;
        SizeT                        rcc_bonded_pt_filter_skipped = 0;
        SizeT                        rcc_bonded_pt_filter_gen = 0;
        // When true, locked PTs are skipped before PT CCD broadphase emission.
        // Default false: removing locked pairs from CCD also removes the last
        // non-penetration guard, so this stays off until the no-penetration
        // scene gate passes (see docs/architecture.md CCD Removal Precondition).
        bool                         rcc_bonded_pt_skip_ccd = false;

        Float reserve_ratio = 1.1;

        BufferDump dump_PTs;
        BufferDump dump_EEs;
        BufferDump dump_PEs;
        BufferDump dump_PPs;

        template <typename T>
        void loose_resize(muda::DeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }
    };

    muda::CBufferView<Vector4i> PTs() const noexcept;
    muda::CBufferView<Vector4i> EEs() const noexcept;
    muda::CBufferView<Vector3i> PEs() const noexcept;
    muda::CBufferView<Vector2i> PPs() const noexcept;
    muda::CBufferView<ActiveVT> VTs() const noexcept;

    muda::CBufferView<Vector4i> friction_PTs() const noexcept;
    muda::CBufferView<Vector4i> friction_EEs() const noexcept;
    muda::CBufferView<Vector3i> friction_PEs() const noexcept;
    muda::CBufferView<Vector2i> friction_PPs() const noexcept;
    muda::CBufferView<ActiveVT> friction_VTs() const noexcept;

    void  set_rcc_bonded_pt_locked_keys(muda::CBufferView<U64> locked_keys) noexcept;
    void  clear_rcc_bonded_pt_locked_keys() noexcept;
    void  set_rcc_bonded_pt_skip_ccd(bool enabled) noexcept;
    SizeT rcc_bonded_pt_filter_skipped_count() const noexcept;
    SizeT rcc_bonded_pt_filter_generation() const noexcept;

    virtual muda::CBufferView<Vector2i> candidate_PTs() const noexcept = 0;
    virtual muda::CBufferView<Vector2i> candidate_EEs() const noexcept = 0;
    virtual muda::CBufferView<Float>    toi_PTs() const noexcept       = 0;
    virtual muda::CBufferView<Float>    toi_EEs() const noexcept       = 0;

  protected:
    virtual void do_build(BuildInfo& info)                = 0;
    virtual void do_detect(DetectInfo& info)              = 0;
    virtual void do_filter_active(FilterActiveInfo& info) = 0;
    virtual void do_filter_toi(FilterTOIInfo& info)       = 0;
    virtual bool do_dump(DumpInfo& info) override;
    virtual bool do_try_recover(RecoverInfo& info) override;
    virtual void do_apply_recover(RecoverInfo& info) override;
    virtual void do_clear_recover(RecoverInfo& info) override;

  private:
    friend class GlobalDCDFilter;
    Impl m_impl;

    virtual void do_build() override final;

    virtual void do_detect(GlobalTrajectoryFilter::DetectInfo& info) override final;
    virtual void do_filter_active(GlobalTrajectoryFilter::FilterActiveInfo& info) override final;
    virtual void do_filter_toi(GlobalTrajectoryFilter::FilterTOIInfo& info) override final;
    virtual void do_record_friction_candidates(
        GlobalTrajectoryFilter::RecordFrictionCandidatesInfo& info) override final;
    virtual void do_label_active_vertices(GlobalTrajectoryFilter::LabelActiveVerticesInfo& info) final override;
    virtual void do_clear_friction_candidates() override final;
};
}  // namespace uipc::backend::cuda
