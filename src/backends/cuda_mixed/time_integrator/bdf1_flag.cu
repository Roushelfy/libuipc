#include <time_integrator/bdf1_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda_mixed::BDF1Flag>
{
  public:
    static U<cuda_mixed::BDF1Flag> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<cuda_mixed::SimEngine&>(engine).world().scene();
        auto itype_attr = scene.config().find<std::string>("integrator/type");

        if(itype_attr->view()[0] != "bdf1")
        {
            return nullptr;  // Not a BDF1 integrator
        }
        return uipc::make_unique<cuda_mixed::BDF1Flag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(BDF1Flag);
void BDF1Flag::do_build() {}
}  // namespace uipc::backend::cuda_mixed
