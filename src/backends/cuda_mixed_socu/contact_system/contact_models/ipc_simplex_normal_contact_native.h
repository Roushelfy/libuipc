#pragma once

#include <contact_system/simplex_normal_contact.h>

namespace uipc::backend::cuda_mixed
{
void assemble_ipc_simplex_normal_contact_native_exact(
    SimplexNormalContact::ContactInfo& info);

void assemble_ipc_simplex_normal_contact_native_exact_PT(
    SimplexNormalContact::ContactInfo& info);
void assemble_ipc_simplex_normal_contact_native_exact_EE(
    SimplexNormalContact::ContactInfo& info);
void assemble_ipc_simplex_normal_contact_native_exact_PE(
    SimplexNormalContact::ContactInfo& info);
void assemble_ipc_simplex_normal_contact_native_exact_PP(
    SimplexNormalContact::ContactInfo& info);
}  // namespace uipc::backend::cuda_mixed
