#include <contact_system/contact_models/ipc_simplex_normal_contact_native.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_normal_contact_native_exact(
    SimplexNormalContact::ContactInfo& info)
{
    assemble_ipc_simplex_normal_contact_native_exact_PT(info);
    assemble_ipc_simplex_normal_contact_native_exact_EE(info);
    assemble_ipc_simplex_normal_contact_native_exact_PE(info);
    assemble_ipc_simplex_normal_contact_native_exact_PP(info);
}
}  // namespace uipc::backend::cuda_mixed
