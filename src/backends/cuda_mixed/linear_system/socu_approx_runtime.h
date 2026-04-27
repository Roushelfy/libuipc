#pragma once

#include <linear_system/socu_approx_report.h>
#include <mixed_precision/policy.h>

#include <cuda_runtime.h>
#include <muda/buffer/device_buffer.h>

#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#include <socu_native/problem_generator.h>
#include <socu_native/solver.h>
#endif

namespace uipc::backend::cuda_mixed
{
template <typename T>
std::string socu_dtype_name()
{
    if constexpr(std::is_same_v<T, float>)
        return "float32";
    else
        return "float64";
}

struct SocuApproxRuntime
{
#if UIPC_WITH_SOCU_NATIVE
    using Scalar = ActivePolicy::SolveScalar;

    struct SolverPlanDeleter
    {
        void operator()(socu_native::SolverPlan* plan) const
        {
            socu_native::destroy_solver_plan(plan);
        }
    };

    socu_native::ProblemShape shape{};
    socu_native::ProblemBufferLayout layout{};
    socu_native::SolverPlanOptions options{};
    std::unique_ptr<socu_native::SolverPlan, SolverPlanDeleter> plan;
    muda::DeviceBuffer<Scalar> device_diag;
    muda::DeviceBuffer<Scalar> device_off_diag;
    muda::DeviceBuffer<Scalar> device_diag_original;
    muda::DeviceBuffer<Scalar> device_off_diag_original;
    muda::DeviceBuffer<Scalar> device_rhs;
    muda::DeviceBuffer<Scalar> device_rhs_original;
    muda::DeviceBuffer<IndexT> device_old_to_chain;
    muda::DeviceBuffer<IndexT> device_chain_to_old;
    muda::DeviceBuffer<double> validation_sums;
    muda::DeviceBuffer<IndexT> report_counters;
    double*                    host_validation_sums = nullptr;
    cudaEvent_t                validation_done = nullptr;
    bool                       mappings_uploaded = false;

    SocuApproxRuntime(const socu_native::ProblemShape& shape_in,
            const socu_native::SolverPlanOptions& options_in)
        : shape(shape_in)
        , layout(socu_native::describe_problem_layout(shape))
        , options(options_in)
    {
    }

    SocuApproxRuntime(const SocuApproxRuntime&) = delete;
    SocuApproxRuntime& operator=(const SocuApproxRuntime&) = delete;

    ~SocuApproxRuntime()
    {
        if(validation_done)
            cudaEventDestroy(validation_done);
        if(host_validation_sums)
            cudaFreeHost(host_validation_sums);
    }

    void reserve(bool debug_validation, bool report_counters_enabled)
    {
        if(device_diag.capacity() < layout.diag_element_count)
            device_diag.reserve(layout.diag_element_count);
        if(device_off_diag.capacity() < layout.off_diag_element_count)
            device_off_diag.reserve(layout.off_diag_element_count);
        if(device_rhs.capacity() < layout.rhs_element_count)
            device_rhs.reserve(layout.rhs_element_count);
        device_diag.resize(layout.diag_element_count);
        device_off_diag.resize(layout.off_diag_element_count);
        device_rhs.resize(layout.rhs_element_count);

        if(device_rhs_original.capacity() < layout.rhs_element_count)
            device_rhs_original.reserve(layout.rhs_element_count);
        if(validation_sums.capacity() < 5)
            validation_sums.reserve(5);
        device_rhs_original.resize(layout.rhs_element_count);
        validation_sums.resize(5);

        if(!host_validation_sums)
        {
            SOCU_NATIVE_CHECK_CUDA(
                cudaHostAlloc(reinterpret_cast<void**>(&host_validation_sums),
                              5 * sizeof(double),
                              cudaHostAllocDefault));
        }
        if(!validation_done)
        {
            SOCU_NATIVE_CHECK_CUDA(cudaEventCreateWithFlags(&validation_done,
                                                            cudaEventDisableTiming));
        }

        if(report_counters_enabled)
        {
            if(report_counters.capacity() < 5)
                report_counters.reserve(5);
            report_counters.resize(5);
        }

        if(debug_validation)
        {
            if(device_diag_original.capacity() < layout.diag_element_count)
                device_diag_original.reserve(layout.diag_element_count);
            if(device_off_diag_original.capacity() < layout.off_diag_element_count)
                device_off_diag_original.reserve(layout.off_diag_element_count);
            device_diag_original.resize(layout.diag_element_count);
            device_off_diag_original.resize(layout.off_diag_element_count);
        }
    }

    void download_validation_sums(cudaStream_t stream)
    {
        SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(host_validation_sums,
                                               validation_sums.data(),
                                               5 * sizeof(double),
                                               cudaMemcpyDeviceToHost,
                                               stream));
        SOCU_NATIVE_CHECK_CUDA(cudaEventRecord(validation_done, stream));
        SOCU_NATIVE_CHECK_CUDA(cudaEventSynchronize(validation_done));
    }

    void upload_mappings_once(const std::vector<IndexT>& old_to_chain,
                              const std::vector<IndexT>& chain_to_old)
    {
        if(mappings_uploaded && device_old_to_chain.size() == old_to_chain.size()
           && device_chain_to_old.size() == chain_to_old.size())
            return;
        device_old_to_chain.resize(old_to_chain.size());
        device_chain_to_old.resize(chain_to_old.size());
        if(!old_to_chain.empty())
            device_old_to_chain.view().copy_from(old_to_chain.data());
        if(!chain_to_old.empty())
            device_chain_to_old.view().copy_from(chain_to_old.data());
        mappings_uploaded = true;
    }

    bool factor_and_solve(cudaStream_t stream)
    {
        const bool created = ensure_plan();
        socu_native::factor_and_solve_inplace_async(
            plan.get(),
            device_diag.data(),
            device_off_diag.data(),
            device_rhs.data(),
            socu_native::LaunchOptions{stream});
        return created;
    }

    bool create_plan() { return ensure_plan(); }

    void snapshot_matrix(cudaStream_t stream)
    {
        const auto diag_bytes = layout.diag_element_count * sizeof(Scalar);
        if(diag_bytes)
        {
            SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_diag_original.data(),
                                                   device_diag.data(),
                                                   diag_bytes,
                                                   cudaMemcpyDeviceToDevice,
                                                   stream));
        }

        const auto off_diag_bytes = layout.off_diag_element_count * sizeof(Scalar);
        if(off_diag_bytes)
        {
            SOCU_NATIVE_CHECK_CUDA(cudaMemcpyAsync(device_off_diag_original.data(),
                                                   device_off_diag.data(),
                                                   off_diag_bytes,
                                                   cudaMemcpyDeviceToDevice,
                                                   stream));
        }
    }

  private:
    bool ensure_plan()
    {
        if(plan != nullptr)
            return false;
        plan.reset(socu_native::create_solver_plan<Scalar>(shape, options));
        return true;
    }
#endif
};


#if UIPC_WITH_SOCU_NATIVE
std::filesystem::path default_mathdx_manifest_path();

std::string to_report_string(socu_native::SolverBackend backend);
std::string to_report_string(socu_native::PerfBackend backend);
std::string to_report_string(socu_native::MathMode mode);
std::string to_report_string(socu_native::GraphMode mode);

template <typename Scalar>
bool validate_mathdx_manifest(const std::filesystem::path& manifest_path,
                              SizeT                       block_size,
                              SocuApproxGateReport&       gate,
                              std::string&                detail);
#endif
}  // namespace uipc::backend::cuda_mixed
