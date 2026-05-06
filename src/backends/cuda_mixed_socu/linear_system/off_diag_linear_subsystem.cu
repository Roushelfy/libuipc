#include <linear_system/off_diag_linear_subsystem.h>
#include <uipc/common/log.h>
#include <fmt/format.h>
namespace uipc::backend::cuda_mixed
{
void OffDiagLinearSubsystem::do_build()
{
    auto&     global_linear_system = require<GlobalLinearSystem>();
    BuildInfo info;
    do_build(info);

    UIPC_ASSERT(info.m_diag_l != nullptr && info.m_diag_r != nullptr,
                "Did you forget to call BuildInfo::connect() in {}'s do_build()?",
                this->name());

    m_l = info.m_diag_l;
    m_r = info.m_diag_r;

    global_linear_system.add_subsystem(this);
}

void OffDiagLinearSubsystem::init()
{
    InitInfo info;
    do_init(info);
}

void OffDiagLinearSubsystem::do_init(InitInfo& info) {}

void OffDiagLinearSubsystem::do_assemble_structured(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    throw SimSystemException{fmt::format(
        "structured_offdiag_subsystem_not_supported: offdiag subsystem '{}' does not support structured Hessian assembly",
        name())};
}

bool OffDiagLinearSubsystem::supports_structured_assembly() const
{
    return do_supports_structured_assembly();
}

void OffDiagLinearSubsystem::assemble_structured(
    GlobalLinearSystem::StructuredAssemblyInfo& info)
{
    do_assemble_structured(info);
}

std::tuple<U64, U64> OffDiagLinearSubsystem::uid() const noexcept
{
    return std::make_tuple(m_l->uid(), m_r->uid());
}
}  // namespace uipc::backend::cuda_mixed
