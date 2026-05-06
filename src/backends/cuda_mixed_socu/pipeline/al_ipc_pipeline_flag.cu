#include <pipeline/al_ipc_pipeline_flag.h>
#include <sim_engine.h>

namespace uipc::backend
{
template <>
class backend::SimSystemCreator<cuda_mixed::ALIPCPipelineFlag>
{
  public:
    static U<cuda_mixed::ALIPCPipelineFlag> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<cuda_mixed::SimEngine&>(engine).world().scene();
        auto ctype_attr = scene.config().find<std::string>("contact/constitution");

        if(ctype_attr->view()[0] != "al-ipc")
        {
            return nullptr;  // Not an AL-IPC pipeline
        }
        return uipc::make_unique<cuda_mixed::ALIPCPipelineFlag>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda_mixed
{
REGISTER_SIM_SYSTEM(ALIPCPipelineFlag);
void ALIPCPipelineFlag::do_build() {}
}  // namespace uipc::backend::cuda_mixed
