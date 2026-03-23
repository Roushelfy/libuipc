#include <time_integrator/bdf2_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda_mixed::BDF2Flag>
{
  public:
    static U<cuda_mixed::BDF2Flag> create(SimEngine& engine)
    {
        auto scene      = dynamic_cast<cuda_mixed::SimEngine&>(engine).world().scene();
        auto itype_attr = scene.config().find<std::string>("integrator/type");

        if(itype_attr->view()[0] != "bdf2")
        {
            return nullptr;  // Not a BDF2 integrator
        }
        return uipc::make_unique<cuda_mixed::BDF2Flag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(BDF2Flag);
void BDF2Flag::do_build() {}
}  // namespace uipc::backend::cuda_mixed
