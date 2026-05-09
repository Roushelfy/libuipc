#pragma once

#include <contact_system/simplex_frictional_contact.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_frictional_contact_native_exact(
    SimplexFrictionalContact::ContactInfo& info);

void assemble_ipc_simplex_frictional_contact_native_exact_PT(
    SimplexFrictionalContact::ContactInfo& info);
void assemble_ipc_simplex_frictional_contact_native_exact_EE(
    SimplexFrictionalContact::ContactInfo& info);
void assemble_ipc_simplex_frictional_contact_native_exact_PE(
    SimplexFrictionalContact::ContactInfo& info);
void assemble_ipc_simplex_frictional_contact_native_exact_PP(
    SimplexFrictionalContact::ContactInfo& info);
}  // namespace uipc::backend::cuda_mixed
