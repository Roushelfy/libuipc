#pragma once

#include <random>

namespace uipc::tensor_core_lab
{
std::mt19937 make_rng(int seed, int sequence, int salt = 0);
double       uniform_real(std::mt19937& rng, double lo, double hi);
double       normal_real(std::mt19937& rng);
}  // namespace uipc::tensor_core_lab
