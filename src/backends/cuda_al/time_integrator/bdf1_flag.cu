#include <time_integrator/bdf1_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda_al::BDF1Flag>
{
  public:
    static U<cuda_al::BDF1Flag> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<cuda_al::SimEngine&>(engine).world().scene();
        if(scene.info()["integrator"]["type"] != "bdf1")
        {
            return nullptr;  // Not a BDF1 integrator
        }
        return uipc::make_unique<cuda_al::BDF1Flag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda_al
{
REGISTER_SIM_SYSTEM(BDF1Flag);
void BDF1Flag::do_build() {}
}  // namespace uipc::backend::cuda_al