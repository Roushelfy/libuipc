#include "rng.h"

namespace uipc::tensor_core_lab
{
std::mt19937 make_rng(int seed, int sequence, int salt)
{
    std::seed_seq seq{seed, sequence, salt, 0x51f15e1d};
    return std::mt19937(seq);
}

double uniform_real(std::mt19937& rng, double lo, double hi)
{
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

double normal_real(std::mt19937& rng)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}
}  // namespace uipc::tensor_core_lab
