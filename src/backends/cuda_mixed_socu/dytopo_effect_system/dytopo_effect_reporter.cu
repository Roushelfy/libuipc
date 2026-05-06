#include <dytopo_effect_system/dytopo_effect_reporter.h>
#include <uipc/common/exception.h>
#include <fmt/format.h>

namespace uipc::backend::cuda_mixed
{
void DyTopoEffectReporter::do_build()
{
    auto& manager = require<GlobalDyTopoEffectManager>();

    BuildInfo info;
    do_build(info);

    manager.add_reporter(this);
}

void DyTopoEffectReporter::init()
{
    InitInfo info;
    do_init(info);
}

void DyTopoEffectReporter::do_init(InitInfo&) {}

void DyTopoEffectReporter::report_energy_extent(GlobalDyTopoEffectManager::EnergyExtentInfo& info)
{
    do_report_energy_extent(info);
}

void DyTopoEffectReporter::report_gradient_hessian_extent(GlobalDyTopoEffectManager::GradientHessianExtentInfo& info)
{
    do_report_gradient_hessian_extent(info);

    UIPC_ASSERT(!(info.gradient_only() && info.m_hessian_count != 0),
                "When gradient_only is true, hessian_count must be 0, but {} provides hessian count={}.\n"
                "Ref: https://github.com/spiriMirror/libuipc/issues/295",
                name(),
                info.m_hessian_count);
}

void DyTopoEffectReporter::assemble(GlobalDyTopoEffectManager::GradientHessianInfo& info)
{
    do_assemble(info);

    m_impl.gradients = info.gradients();
    m_impl.hessians  = info.hessians();
}

bool DyTopoEffectReporter::do_supports_structured_hessian() const
{
    return false;
}

void DyTopoEffectReporter::do_assemble_structured_hessian(
    GlobalDyTopoEffectManager::StructuredHessianInfo&)
{
    throw Exception{fmt::format(
        "structured_dytopo_reporter_not_supported: reporter '{}' cannot write "
        "GradientStructuredHessian directly to the structured sink",
        name())};
}

bool DyTopoEffectReporter::supports_structured_hessian() const
{
    return do_supports_structured_hessian();
}

void DyTopoEffectReporter::assemble_structured_hessian(
    GlobalDyTopoEffectManager::StructuredHessianInfo& info)
{
    do_assemble_structured_hessian(info);
}

void DyTopoEffectReporter::compute_energy(GlobalDyTopoEffectManager::EnergyInfo& info)
{
    do_compute_energy(info);

    m_impl.energies = info.energies();
}
}  // namespace uipc::backend::cuda_mixed
