#pragma once

#include <contact_system/simplex_normal_contact.h>
#include <linear_system/socu_native_contact_assembly_sink.h>
#include <linear_system/socu_native_contact_targets.h>

namespace uipc::backend::cuda_mixed
{
struct SimplexNormalContactNativeContext
{
    SocuNativeContactAssemblySink<SimplexNormalContact::StoreScalar,
                                  ActivePolicy::SolveScalar>
        sink;
    muda::CBufferView<SocuNativeContactStencilTarget> PT_targets;
    muda::CBufferView<SocuNativeContactStencilTarget> EE_targets;
    muda::CBufferView<SocuNativeContactStencilTarget> PE_targets;
    muda::CBufferView<SocuNativeContactStencilTarget> PP_targets;
};

void assemble_ipc_simplex_normal_contact_native_exact(
    SimplexNormalContact::ContactInfo&          info,
    const SimplexNormalContactNativeContext& native_context);

void assemble_ipc_simplex_normal_contact_native_exact_PT(
    SimplexNormalContact::ContactInfo&          info,
    const SimplexNormalContactNativeContext& native_context);
void assemble_ipc_simplex_normal_contact_native_exact_EE(
    SimplexNormalContact::ContactInfo&          info,
    const SimplexNormalContactNativeContext& native_context);
void assemble_ipc_simplex_normal_contact_native_exact_PE(
    SimplexNormalContact::ContactInfo&          info,
    const SimplexNormalContactNativeContext& native_context);
void assemble_ipc_simplex_normal_contact_native_exact_PP(
    SimplexNormalContact::ContactInfo&          info,
    const SimplexNormalContactNativeContext& native_context);
}  // namespace uipc::backend::cuda_mixed
