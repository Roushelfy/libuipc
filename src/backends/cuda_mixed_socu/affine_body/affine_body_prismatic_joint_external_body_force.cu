#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/constraints/affine_body_prismatic_joint_external_body_force_constraint.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/constitutions/affine_body_prismatic_joint_function.h>
#include <time_integrator/time_integrator.h>
#include <muda/ext/eigen/atomic.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class AffineBodyPrismaticJointExternalBodyForce final : public AffineBodyExternalForceReporter
{
  public:
    static constexpr U64 UID = 667;

    using AffineBodyExternalForceReporter::AffineBodyExternalForceReporter;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    SimSystemSlot<AffineBodyPrismaticJointExternalBodyForceConstraint> constraint;

    virtual void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
        constraint = require<AffineBodyPrismaticJointExternalBodyForceConstraint>();
    }

    U64 get_uid() const noexcept override { return UID; }

    void do_init() override
    {
        // Nothing to do
    }

    void do_step(ExternalForceInfo& info) override
    {
        SizeT force_count = constraint->forces().size();
        if(force_count == 0)
            return;

        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(force_count,
                   [external_forces = info.external_forces().viewer().name("external_forces"),
                    body_ids = constraint->body_ids().viewer().name("body_ids"),
                    forces   = constraint->forces().viewer().name("forces"),
                    rest_tangents = constraint->rest_tangents().viewer().name("rest_tangents"),
                    qs = affine_body_dynamics->qs().cviewer().name("qs")] __device__(int i) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;
                       Vector2i bids = body_ids(i);
                       Alu      f    = safe_cast<Alu>(forces(i));

                       Eigen::Matrix<Alu, 6, 1> t_bar =
                           rest_tangents(i).template cast<Alu>();

                       Eigen::Matrix<Alu, 12, 1> q_i =
                           qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> q_j =
                           qs(bids(1)).template cast<Alu>();

                       auto vec_x = [](const Eigen::Matrix<Alu, 3, 1>&  x_bar,
                                       const Eigen::Matrix<Alu, 12, 1>& q)
                       {
                           Eigen::Matrix<Alu, 3, 1> v;
                           v[0] = x_bar.dot(q.template segment<3>(3));
                           v[1] = x_bar.dot(q.template segment<3>(6));
                           v[2] = x_bar.dot(q.template segment<3>(9));
                           return v;
                       };

                       Eigen::Matrix<Alu, 3, 1> t_i = vec_x(t_bar.segment<3>(0), q_i);
                       Eigen::Matrix<Alu, 3, 1> t_j = vec_x(t_bar.segment<3>(3), q_j);

                       // Build 12D force vectors per doc: +f*t to body_i, -f*t to body_j
                       // F = [fx, fy, fz, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                       Eigen::Matrix<Alu, 12, 1> F_i_alu =
                           Eigen::Matrix<Alu, 12, 1>::Zero();
                       F_i_alu.segment<3>(0) = f * t_i;

                       Eigen::Matrix<Alu, 12, 1> F_j_alu =
                           Eigen::Matrix<Alu, 12, 1>::Zero();
                       F_j_alu.segment<3>(0) = -f * t_j;

                       // Scatter add to external forces
                       eigen::atomic_add(external_forces(bids(0)),
                                         downcast_gradient<Float>(F_i_alu));
                       eigen::atomic_add(external_forces(bids(1)),
                                         downcast_gradient<Float>(F_j_alu));
                   });
    }
};

REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointExternalBodyForce);

class AffineBodyPrismaticJointExternalForceTimeIntegrator : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<AffineBodyPrismaticJointExternalBodyForceConstraint> constraint;
    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    void do_init(InitInfo& info) override {}

    void do_build(BuildInfo& info) override
    {
        constraint          = require<AffineBodyPrismaticJointExternalBodyForceConstraint>();
        affine_body_dynamics = require<AffineBodyDynamics>();
    }

    void do_predict_dof(PredictDofInfo& info) override {}

    void do_update_state(UpdateVelocityInfo& info) override
    {
        using namespace muda;
        namespace DPJ = sym::affine_body_driving_prismatic_joint;

        SizeT N = constraint->body_ids().size();
        if(N == 0)
            return;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(N,
                   [body_ids       = constraint->body_ids().cviewer().name("body_ids"),
                    rest_positions = constraint->rest_positions().cviewer().name("rest_positions"),
                    rest_tangents  = constraint->rest_tangents().cviewer().name("rest_tangents"),
                    init_distances = constraint->init_distances().cviewer().name("init_distances"),
                    current_distances = constraint->current_distances().viewer().name("current_distances"),
                    qs = affine_body_dynamics->qs().cviewer().name("qs")] __device__(int I)
                   {
                       using Alu = ActivePolicy::AluScalar;
                       Vector2i bids = body_ids(I);

                       Eigen::Matrix<Alu, 12, 1> q_i =
                           qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> q_j =
                           qs(bids(1)).template cast<Alu>();

                       Eigen::Matrix<Alu, 6, 1> C_bar =
                           rest_positions(I).template cast<Alu>();
                       Eigen::Matrix<Alu, 6, 1> t_bar =
                           rest_tangents(I).template cast<Alu>();

                       Eigen::Matrix<Alu, 9, 1> F01_q;
                       DPJ::F01_q<Alu>(F01_q,
                                       C_bar.segment<3>(0),
                                       t_bar.segment<3>(0),
                                       q_i,
                                       C_bar.segment<3>(3),
                                       t_bar.segment<3>(3),
                                       q_j);

                       Alu distance;
                       DPJ::Distance<Alu>(distance, F01_q);

                       current_distances(I) = safe_cast<Float>(
                           distance - safe_cast<Alu>(init_distances(I)));
                   });
    }
};
REGISTER_SIM_SYSTEM(AffineBodyPrismaticJointExternalForceTimeIntegrator);

}  // namespace uipc::backend::cuda_mixed
