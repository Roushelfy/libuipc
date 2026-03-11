#include <energy_component_flags.h>

namespace uipc::backend::cuda_mixed
{
std::string enum_flags_name(EnergyComponentFlags flags)
{
    return magic_enum::enum_flags_name(flags);
}
}  // namespace uipc::backend::cuda_mixed
