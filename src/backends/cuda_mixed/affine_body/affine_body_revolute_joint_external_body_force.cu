#include <numbers>
#include <affine_body/affine_body_external_force_reporter.h>
#include <affine_body/constraints/affine_body_revolute_joint_external_body_force_constraint.h>
#include <affine_body/affine_body_dynamics.h>
#include <affine_body/constitutions/affine_body_revolute_joint_function.h>
#include <time_integrator/time_integrator.h>
#include <muda/ext/eigen/atomic.h>
#include <mixed_precision/policy.h>
#include <mixed_precision/cast.h>

namespace uipc::backend::cuda_mixed
{
class AffineBodyRevoluteJointExternalBodyForce final : public AffineBodyExternalForceReporter
{
  public:
    static constexpr U64 UID = 668;

    using AffineBodyExternalForceReporter::AffineBodyExternalForceReporter;

    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
    SimSystemSlot<AffineBodyRevoluteJointExternalBodyForceConstraint> constraint;

    virtual void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = require<AffineBodyDynamics>();
        constraint = require<AffineBodyRevoluteJointExternalBodyForceConstraint>();
    }

    U64 get_uid() const noexcept override { return UID; }

    void do_init() override {}

    void do_step(ExternalForceInfo& info) override
    {
        SizeT torque_count = constraint->torques().size();
        if(torque_count == 0)
            return;

        using namespace muda;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(torque_count,
                   [external_forces = info.external_forces().viewer().name("external_forces"),
                    body_ids = constraint->body_ids().viewer().name("body_ids"),
                    torques  = constraint->torques().viewer().name("torques"),
                    rest_positions = constraint->rest_positions().viewer().name("rest_positions"),
                    qs = affine_body_dynamics->qs().cviewer().name("qs")] __device__(int i) mutable
                   {
                       using Alu = ActivePolicy::AluScalar;
                       Vector2i bids = body_ids(i);
                       Alu      tau  = safe_cast<Alu>(torques(i));

                       Eigen::Matrix<Alu, 12, 1> X_bar =
                           rest_positions(i).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> q_i =
                           qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> q_j =
                           qs(bids(1)).template cast<Alu>();

                       Eigen::Matrix<Alu, 3, 1> x0_bar = X_bar.segment<3>(0);
                       Eigen::Matrix<Alu, 3, 1> x1_bar = X_bar.segment<3>(3);
                       Eigen::Matrix<Alu, 3, 1> x2_bar = X_bar.segment<3>(6);
                       Eigen::Matrix<Alu, 3, 1> x3_bar = X_bar.segment<3>(9);

                       auto vec_x = [](const Eigen::Matrix<Alu, 3, 1>&  x_bar,
                                       const Eigen::Matrix<Alu, 12, 1>& q)
                       {
                           Eigen::Matrix<Alu, 3, 1> v;
                           v[0] = x_bar.dot(q.template segment<3>(3));
                           v[1] = x_bar.dot(q.template segment<3>(6));
                           v[2] = x_bar.dot(q.template segment<3>(9));
                           return v;
                       };

                       // Axis direction in world frame
                       Eigen::Matrix<Alu, 3, 1> e_world_i =
                           vec_x(x1_bar - x0_bar, q_i).normalized();
                       Eigen::Matrix<Alu, 3, 1> e_world_j =
                           vec_x(x3_bar - x2_bar, q_j).normalized();

                       // Vector from axis point x0 to center of mass (x_bar=0) in world:
                       //   c - x0_world = c - (c + A * x0_bar) = -A * x0_bar
                       Eigen::Matrix<Alu, 3, 1> d_i = -vec_x(x0_bar, q_i);
                       Eigen::Matrix<Alu, 3, 1> d_j = -vec_x(x2_bar, q_j);

                       // Project center of mass onto axis, lever arm is the
                       // perpendicular component: r = d - (d·e)*e
                       Eigen::Matrix<Alu, 3, 1> r_i =
                           d_i - d_i.dot(e_world_i) * e_world_i;
                       Eigen::Matrix<Alu, 3, 1> r_j =
                           d_j - d_j.dot(e_world_j) * e_world_j;

                       Alu r_sq_i = r_i.squaredNorm();
                       Alu r_sq_j = r_j.squaredNorm();

                       // Tangential force at center of mass:
                       //   F = tau * (e × r) / |r|^2
                       // Body_i receives -tau (reaction), body_j receives +tau.

                       constexpr Alu eps = static_cast<Alu>(1e-12);

                       Eigen::Matrix<Alu, 12, 1> F_i_alu =
                           Eigen::Matrix<Alu, 12, 1>::Zero();
                       if(r_sq_i > eps)
                       {
                           F_i_alu.segment<3>(0) =
                               -tau * e_world_i.cross(r_i) / r_sq_i;
                       }

                       Eigen::Matrix<Alu, 12, 1> F_j_alu =
                           Eigen::Matrix<Alu, 12, 1>::Zero();
                       if(r_sq_j > eps)
                       {
                           F_j_alu.segment<3>(0) =
                               tau * e_world_j.cross(r_j) / r_sq_j;
                       }

                       eigen::atomic_add(external_forces(bids(0)),
                                         downcast_gradient<Float>(F_i_alu));
                       eigen::atomic_add(external_forces(bids(1)),
                                         downcast_gradient<Float>(F_j_alu));
                   });
    }
};

REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointExternalBodyForce);

class AffineBodyRevoluteJointExternalForceTimeIntegrator : public TimeIntegrator
{
  public:
    using TimeIntegrator::TimeIntegrator;

    SimSystemSlot<AffineBodyRevoluteJointExternalBodyForceConstraint> constraint;
    SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;

    void do_init(InitInfo& info) override {}

    void do_build(BuildInfo& info) override
    {
        constraint = require<AffineBodyRevoluteJointExternalBodyForceConstraint>();
        affine_body_dynamics = require<AffineBodyDynamics>();
    }

    void do_predict_dof(PredictDofInfo& info) override {}

    void do_update_state(UpdateVelocityInfo& info) override
    {
        using namespace muda;
        namespace DRJ = sym::affine_body_driving_revolute_joint;

        SizeT N = constraint->body_ids().size();
        if(N == 0)
            return;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(N,
                   [body_ids = constraint->body_ids().cviewer().name("body_ids"),
                    rest_positions = constraint->rest_positions().cviewer().name("rest_positions"),
                    init_angles = constraint->init_angles().cviewer().name("init_angles"),
                    current_angles = constraint->current_angles().viewer().name("current_angles"),
                    qs = affine_body_dynamics->qs().cviewer().name("qs")] __device__(int I)
                   {
                       using Alu = ActivePolicy::AluScalar;
                       constexpr Alu PI_alu = static_cast<Alu>(std::numbers::pi_v<double>);

                       Vector2i bids = body_ids(I);

                       Eigen::Matrix<Alu, 12, 1> q_i =
                           qs(bids(0)).template cast<Alu>();
                       Eigen::Matrix<Alu, 12, 1> q_j =
                           qs(bids(1)).template cast<Alu>();

                       Eigen::Matrix<Alu, 12, 1> X_bar =
                           rest_positions(I).template cast<Alu>();

                       Eigen::Matrix<Alu, 3, 1> x0_bar = X_bar.segment<3>(0);
                       Eigen::Matrix<Alu, 3, 1> x1_bar = X_bar.segment<3>(3);
                       Eigen::Matrix<Alu, 3, 1> x2_bar = X_bar.segment<3>(6);
                       Eigen::Matrix<Alu, 3, 1> x3_bar = X_bar.segment<3>(9);

                       Eigen::Matrix<Alu, 3, 1> h_bar_i =
                           (x1_bar - x0_bar) * static_cast<Alu>(0.5);
                       Eigen::Matrix<Alu, 3, 1> h_bar_j =
                           (x3_bar - x2_bar) * static_cast<Alu>(0.5);

                       Eigen::Matrix<Alu, 3, 1> e_bar_i = h_bar_i.normalized();
                       Eigen::Matrix<Alu, 3, 1> e_bar_j = h_bar_j.normalized();

                       auto toNormal =
                           [](const Eigen::Matrix<Alu, 3, 1>& W) -> Eigen::Matrix<Alu, 3, 1>
                       {
                           Eigen::Matrix<Alu, 3, 1> ref =
                               abs(W.dot(Eigen::Matrix<Alu, 3, 1>(1, 0, 0)))
                                       < static_cast<Alu>(0.99) ?
                                   Eigen::Matrix<Alu, 3, 1>(1, 0, 0) :
                                   Eigen::Matrix<Alu, 3, 1>(0, 1, 0);
                           Eigen::Matrix<Alu, 3, 1> U = ref.cross(W).normalized();
                           return W.cross(U).normalized();
                       };

                       Eigen::Matrix<Alu, 3, 1> n_bar_i = toNormal(e_bar_i);
                       Eigen::Matrix<Alu, 3, 1> v_bar_i =
                           n_bar_i.cross(e_bar_i).normalized();
                       Eigen::Matrix<Alu, 3, 1> n_bar_j = toNormal(e_bar_j);
                       Eigen::Matrix<Alu, 3, 1> v_bar_j =
                           n_bar_j.cross(e_bar_j).normalized();

                       Eigen::Matrix<Alu, 12, 1> F01_q;
                       DRJ::F01_q<Alu>(F01_q, v_bar_i, n_bar_i, q_i, v_bar_j, n_bar_j, q_j);

                       Alu curr_angle;
                       DRJ::currAngle<Alu>(curr_angle, F01_q);

                       auto map2range = [=](Alu angle) -> Alu
                       {
                           if(angle > PI_alu)
                               angle -= static_cast<Alu>(2) * PI_alu;
                           else if(angle < -PI_alu)
                               angle += static_cast<Alu>(2) * PI_alu;
                           return angle;
                       };

                       current_angles(I) = safe_cast<Float>(
                           map2range(curr_angle - safe_cast<Alu>(init_angles(I))));
                   });
    }
};
REGISTER_SIM_SYSTEM(AffineBodyRevoluteJointExternalForceTimeIntegrator);

}  // namespace uipc::backend::cuda_mixed
