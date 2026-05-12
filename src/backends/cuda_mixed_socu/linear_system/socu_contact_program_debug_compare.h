#pragma once

#include <linear_system/socu_contact_assembly_plan.h>

namespace uipc::backend::cuda_mixed
{
struct SocuContactProgramDebugExpectation
{
    SocuContactProgramKind program_kind = SocuContactProgramKind::Skipped;
    std::uint16_t task_count = 0;
};

struct SocuContactProgramDebugCompare
{
    SocuContactAssemblyPlanView plan;

    MUDA_DEVICE bool matches(SocuContactSourceId source_id,
                             SizeT local_contact_id,
                             SocuContactProgramDebugExpectation expected) const noexcept
    {
        if(!plan.valid())
            return false;

        const auto mapping = plan.program_for(source_id, local_contact_id);
        if(mapping.status != SocuContactProgramMapStatus::Valid
           || mapping.program_id == SocuInvalidContactProgramId
           || static_cast<SizeT>(mapping.program_id) >= plan.programs.size())
            return expected.program_kind == SocuContactProgramKind::Skipped
                   && expected.task_count == 0;

        const auto program =
            plan.programs.data()[static_cast<SizeT>(mapping.program_id)];
        return program.program_kind == expected.program_kind
               && program.task_count == expected.task_count;
    }
};
}  // namespace uipc::backend::cuda_mixed
