#include <contact_system/contact_models/ipc_simplex_normal_contact_native.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_normal_contact_native_exact(
    SimplexNormalContact::ContactInfo& info,
    const SimplexNormalContactNativeContext& native_context)
{
    assemble_ipc_simplex_normal_contact_native_exact_PT(info, native_context);
    assemble_ipc_simplex_normal_contact_native_exact_EE(info, native_context);
    assemble_ipc_simplex_normal_contact_native_exact_PE(info, native_context);
    assemble_ipc_simplex_normal_contact_native_exact_PP(info, native_context);
}
}  // namespace uipc::backend::cuda_mixed
