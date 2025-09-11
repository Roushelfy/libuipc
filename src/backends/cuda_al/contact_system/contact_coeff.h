#pragma once
#include <type_define.h>

namespace uipc::backend::cuda_al
{
class ContactCoeff
{
  public:
    // normal stiffness
    Float kappa = 0.0;
    // friction coefficient
    Float mu = 0.0;
};
}  // namespace uipc::backend::cuda_al
