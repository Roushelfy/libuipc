#include <inter_primitive_effect_system/inter_primitive_constitution.h>
#include <fmt/format.h>

namespace uipc::backend::cuda_mixed
{
void InterPrimitiveConstitution::do_build()
{
    auto all_uids = world().scene().constitution_tabular().uids();
    if(!std::binary_search(all_uids.begin(), all_uids.end(), uid()))
    {
        throw SimSystemException(
            fmt::format("{} requires Constraint UID={}", name(), uid()));
    }

    auto& manager = require<InterPrimitiveConstitutionManager>();

    // let the subclass take care of its own build
    BuildInfo info;
    do_build(info);

    manager.add_constitution(this);
}

void InterPrimitiveConstitution::init(FilteredInfo& info)
{
    do_init(info);
}

void InterPrimitiveConstitution::report_energy_extent(EnergyExtentInfo& info)
{
    do_report_energy_extent(info);
}

void InterPrimitiveConstitution::compute_energy(ComputeEnergyInfo& info)
{
    do_compute_energy(info);
}

void InterPrimitiveConstitution::report_gradient_hessian_extent(GradientHessianExtentInfo& info)
{
    do_report_gradient_hessian_extent(info);

    UIPC_ASSERT(!(info.gradient_only() && info.m_hessian_count != 0),
                "When gradient_only is true, hessian_count must be 0, but {} provides hessian count={}.\n"
                "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                name(),
                info.m_hessian_count);
}

void InterPrimitiveConstitution::compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    do_compute_gradient_hessian(info);
}

bool InterPrimitiveConstitution::do_supports_structured_hessian() const
{
    return false;
}

void InterPrimitiveConstitution::do_compute_structured_hessian(StructuredHessianInfo&)
{
    throw SimSystemException{fmt::format(
        "structured_inter_primitive_constitution_not_supported: constitution '{}' cannot write "
        "GradientStructuredHessian directly to the structured sink",
        name())};
}

bool InterPrimitiveConstitution::supports_structured_hessian() const
{
    return do_supports_structured_hessian();
}

void InterPrimitiveConstitution::compute_structured_hessian(StructuredHessianInfo& info)
{
    do_compute_structured_hessian(info);
}

U64 InterPrimitiveConstitution::uid() const noexcept
{
    return get_uid();
}
}  // namespace uipc::backend::cuda_mixed
