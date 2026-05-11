#pragma once

#include <contact_system/simplex_frictional_contact.h>
#include <linear_system/socu_native_contact_assembly_sink.h>
#include <linear_system/socu_native_contact_targets.h>

namespace uipc::backend::cuda_mixed
{
struct SimplexFrictionalContactNativeContext
{
    SocuNativeContactAssemblySink<SimplexFrictionalContact::StoreScalar,
                                  ActivePolicy::SolveScalar>
        sink;
    muda::CBufferView<SocuNativeContactStencilTarget> PT_targets;
    muda::CBufferView<SocuNativeContactStencilTarget> EE_targets;
    muda::CBufferView<SocuNativeContactStencilTarget> PE_targets;
    muda::CBufferView<SocuNativeContactStencilTarget> PP_targets;
};

void assemble_ipc_simplex_frictional_contact_native_exact(
    SimplexFrictionalContact::ContactInfo&          info,
    const SimplexFrictionalContactNativeContext& native_context);

void assemble_ipc_simplex_frictional_contact_native_exact_PT(
    SimplexFrictionalContact::ContactInfo&          info,
    const SimplexFrictionalContactNativeContext& native_context);
void assemble_ipc_simplex_frictional_contact_native_exact_EE(
    SimplexFrictionalContact::ContactInfo&          info,
    const SimplexFrictionalContactNativeContext& native_context);
void assemble_ipc_simplex_frictional_contact_native_exact_PE(
    SimplexFrictionalContact::ContactInfo&          info,
    const SimplexFrictionalContactNativeContext& native_context);
void assemble_ipc_simplex_frictional_contact_native_exact_PP(
    SimplexFrictionalContact::ContactInfo&          info,
    const SimplexFrictionalContactNativeContext& native_context);
}  // namespace uipc::backend::cuda_mixed
