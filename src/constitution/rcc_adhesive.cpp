#include <uipc/constitution/rcc_adhesive.h>
#include <uipc/builtin/constitution_uid_auto_register.h>
#include <uipc/core/contact_model_collection.h>
#include <uipc/geometry/attribute_collection.h>
#include <uipc/geometry/attribute_slot.h>
#include <uipc/common/log.h>

namespace uipc::constitution
{
constexpr U64 RCCAdhesiveUID = 1000ull;

REGISTER_CONSTITUTION_UIDS()
{
    using namespace uipc::builtin;
    list<UIDInfo> uids;
    uids.push_back(UIDInfo{.uid = RCCAdhesiveUID,
                           .name = "RCCAdhesive",
                           .type = "ContactModel"});
    return uids;
}

RCCAdhesive::RCCAdhesive(const Json& config) noexcept
    : m_config(config)
{
}

Json RCCAdhesive::default_config() noexcept
{
    return Json::object();
}

U64 RCCAdhesive::get_uid() const noexcept
{
    return RCCAdhesiveUID;
}

void RCCAdhesive::apply_to(core::ContactTabular& tabular) const
{
    auto models = tabular.contact_models();

    // Idempotent: create<T>() returns the existing slot when the attribute is
    // already present, so calling apply_to twice is safe.
    if(!models.find<Float>("Cn"))
        models.create<Float>("Cn", Float{0});
    if(!models.find<Float>("Ct"))
        models.create<Float>("Ct", Float{0});
    if(!models.find<Float>("W"))
        models.create<Float>("W", Float{0});
    if(!models.find<Float>("eta"))
        models.create<Float>("eta", Float{1});
    if(!models.find<Float>("bonding_rate"))
        models.create<Float>("bonding_rate", Float{0});
    if(!models.find<Float>("p0"))
        models.create<Float>("p0", Float{0});
    if(!models.find<Float>("initial_beta"))
        models.create<Float>("initial_beta", Float{0});
    if(!models.find<IndexT>("adhesion_enabled"))
        models.create<IndexT>("adhesion_enabled", IndexT{0});
}

static IndexT _find_pair_index(core::ContactTabular&       tabular,
                               const core::ContactElement& L,
                               const core::ContactElement& R)
{
    auto models = tabular.contact_models();
    auto topo   = models.find<Vector2i>("topo");
    UIPC_ASSERT(topo,
                "ContactTabular has no 'topo' attribute. ContactTabular should "
                "always create it on construction.");

    Vector2i ids{L.id(), R.id()};
    if(ids.x() > ids.y())
        std::swap(ids.x(), ids.y());

    auto view = topo->view();
    for(IndexT i = 0; i < static_cast<IndexT>(view.size()); ++i)
    {
        const auto& t = view[i];
        if(t.x() == ids.x() && t.y() == ids.y())
            return i;
    }
    return -1;
}

static void _write_row(core::ContactTabular& tabular,
                       IndexT                index,
                       Float                 Cn,
                       Float                 Ct,
                       Float                 W,
                       Float                 eta,
                       Float                 bonding_rate,
                       Float                 p0,
                       Float                 initial_beta,
                       bool                  enabled)
{
    auto models = tabular.contact_models();

    auto cn_attr  = models.find<Float>("Cn");
    auto ct_attr  = models.find<Float>("Ct");
    auto w_attr   = models.find<Float>("W");
    auto eta_attr = models.find<Float>("eta");
    auto r_attr   = models.find<Float>("bonding_rate");
    auto p0_attr  = models.find<Float>("p0");
    auto ib_attr  = models.find<Float>("initial_beta");
    auto en_attr  = models.find<IndexT>("adhesion_enabled");

    UIPC_ASSERT(cn_attr && ct_attr && w_attr && eta_attr && r_attr && p0_attr
                    && ib_attr && en_attr,
                "RCCAdhesive attributes are missing on ContactTabular. Did you "
                "forget to call RCCAdhesive::apply_to(tabular) first?");

    geometry::view(*cn_attr)[index]  = Cn;
    geometry::view(*ct_attr)[index]  = Ct;
    geometry::view(*w_attr)[index]   = W;
    geometry::view(*eta_attr)[index] = eta;
    geometry::view(*r_attr)[index]   = bonding_rate;
    geometry::view(*p0_attr)[index]  = p0;
    geometry::view(*ib_attr)[index]  = initial_beta;
    geometry::view(*en_attr)[index]  = enabled ? IndexT{1} : IndexT{0};
}

void RCCAdhesive::set(core::ContactTabular&       tabular,
                      const core::ContactElement& L,
                      const core::ContactElement& R,
                      Float                       Cn,
                      Float                       Ct,
                      Float                       W,
                      Float                       eta,
                      Float                       bonding_rate,
                      Float                       p0,
                      Float                       initial_beta,
                      bool                        enabled) const
{
    apply_to(tabular);

    IndexT idx = _find_pair_index(tabular, L, R);
    UIPC_ASSERT(idx >= 0,
                "RCCAdhesive::set: contact pair (L={}, R={}) is not present in "
                "the ContactTabular. Call tabular.insert(L, R, friction_rate, "
                "resistance) first.",
                L.id(),
                R.id());

    _write_row(tabular, idx, Cn, Ct, W, eta, bonding_rate, p0, initial_beta, enabled);
}

void RCCAdhesive::default_model(core::ContactTabular& tabular,
                                Float                 Cn,
                                Float                 Ct,
                                Float                 W,
                                Float                 eta,
                                Float                 bonding_rate,
                                Float                 p0,
                                Float                 initial_beta,
                                bool                  enabled) const
{
    apply_to(tabular);
    // ContactTabular always has a default model at row 0.
    _write_row(tabular, 0, Cn, Ct, W, eta, bonding_rate, p0, initial_beta, enabled);
}
}  // namespace uipc::constitution
