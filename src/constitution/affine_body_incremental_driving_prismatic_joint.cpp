#include <uipc/constitution/affine_body_incremental_driving_prismatic_joint.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/builtin/constitution_type.h>
#include <uipc/builtin/attribute_name.h>

namespace uipc::constitution
{
static constexpr U64 ConstitutionUID   = 34;
static constexpr U64 PrismaticJointUID = 20;  // UID of AffineBodyPrismaticJoint

static constexpr std::string_view PDIsConstrainedName = "pd/is_constrained";
static constexpr std::string_view PDStrengthName      = "pd/strength";
static constexpr std::string_view PDAimIncrementName  = "pd/aim_increment";

REGISTER_CONSTITUTION_UIDS()
{
    using namespace uipc::builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid  = ConstitutionUID,
                           .name = "AffineBodyIncrementalDrivingPrismaticJoint",
                           .type = string{builtin::Constraint}});
    return uids;
}

AffineBodyIncrementalDrivingPrismaticJoint::AffineBodyIncrementalDrivingPrismaticJoint(const Json& config)
{
    m_config = config;
}

AffineBodyIncrementalDrivingPrismaticJoint::~AffineBodyIncrementalDrivingPrismaticJoint() = default;

void AffineBodyIncrementalDrivingPrismaticJoint::apply_to(geometry::SimplicialComplex& sc)
{
    UIPC_ASSERT_THROW(sc.dim() == 1,
                "AffineBodyIncrementalDrivingPrismaticJoint can only be applied to 1D simplicial complex (linemesh), "
                "but got {}D",
                sc.dim());

    Base::apply_to(sc);

    auto uid = sc.meta().find<U64>(builtin::constitution_uid);
    UIPC_ASSERT_THROW(uid && uid->view()[0] == PrismaticJointUID,
                "Simplicial complex does not have constitution uid {}. "
                "Please apply an AffineBodyPrismaticJoint before applying AffineBodyIncrementalDrivingPrismaticJoint",
                PrismaticJointUID);

    auto is_constrained = sc.edges().find<IndexT>(PDIsConstrainedName);
    if(!is_constrained)
    {
        is_constrained = sc.edges().create<IndexT>(PDIsConstrainedName, 0);
    }
    auto is_constrained_view = view(*is_constrained);
    std::ranges::fill(is_constrained_view, 0);

    auto strength = sc.edges().find<Float>(PDStrengthName);
    if(!strength)
    {
        strength = sc.edges().create<Float>(PDStrengthName, 0.0);
    }
    auto strength_view = view(*strength);
    std::ranges::fill(strength_view, 0.0);

    auto aim_increment = sc.edges().find<Float>(PDAimIncrementName);
    if(!aim_increment)
    {
        aim_increment = sc.edges().create<Float>(PDAimIncrementName, 0.0);
    }
    auto aim_increment_view = view(*aim_increment);
    std::ranges::fill(aim_increment_view, 0.0);
}

Json AffineBodyIncrementalDrivingPrismaticJoint::default_config()
{
    return Json::object();
}

U64 AffineBodyIncrementalDrivingPrismaticJoint::get_uid() const noexcept
{
    return ConstitutionUID;
}

}  // namespace uipc::constitution
