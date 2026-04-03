#pragma once
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <tcl/device_info.h>
#include <tcl/mode.h>

namespace uipc::tensor_core_lab
{
template <typename T>
class DeviceBuffer
{
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { resize(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : m_ptr(other.m_ptr)
        , m_size(other.m_size)
    {
        other.m_ptr  = nullptr;
        other.m_size = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if(this != &other)
        {
            reset();
            m_ptr       = other.m_ptr;
            m_size      = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    void resize(size_t count)
    {
        if(count == m_size)
            return;
        reset();
        if(count == 0)
            return;
        if(cudaMalloc(reinterpret_cast<void**>(&m_ptr), count * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");
        m_size = count;
    }

    void copy_from_host(std::span<const T> host)
    {
        if(host.size() != m_size)
            throw std::runtime_error("copy_from_host size mismatch");
        if(cudaMemcpy(m_ptr,
                      host.data(),
                      m_size * sizeof(T),
                      cudaMemcpyHostToDevice)
           != cudaSuccess)
            throw std::runtime_error("cudaMemcpy H2D failed");
    }

    void copy_to_host(std::span<T> host) const
    {
        if(host.size() != m_size)
            throw std::runtime_error("copy_to_host size mismatch");
        if(cudaMemcpy(host.data(),
                      m_ptr,
                      m_size * sizeof(T),
                      cudaMemcpyDeviceToHost)
           != cudaSuccess)
            throw std::runtime_error("cudaMemcpy D2H failed");
    }

    [[nodiscard]] T* data() noexcept { return m_ptr; }
    [[nodiscard]] const T* data() const noexcept { return m_ptr; }
    [[nodiscard]] size_t size() const noexcept { return m_size; }

  private:
    void reset() noexcept
    {
        if(m_ptr)
            cudaFree(m_ptr);
        m_ptr  = nullptr;
        m_size = 0;
    }

    T*     m_ptr  = nullptr;
    size_t m_size = 0;
};

enum class ImplPath
{
    Baseline = 0,
    TcBlas   = 1,
    TcWmma   = 2,
};

enum class TensorCoreVerification
{
    No                   = 0,
    Yes                  = 1,
    BlockedByPermissions = 2,
};

constexpr const char* to_string(ImplPath path) noexcept
{
    switch(path)
    {
        case ImplPath::Baseline:
            return "baseline";
        case ImplPath::TcBlas:
            return "tc_blas";
        case ImplPath::TcWmma:
            return "tc_wmma";
    }
    return "baseline";
}

constexpr const char* to_string(TensorCoreVerification verification) noexcept
{
    switch(verification)
    {
        case TensorCoreVerification::No:
            return "no";
        case TensorCoreVerification::Yes:
            return "yes";
        case TensorCoreVerification::BlockedByPermissions:
            return "blocked_by_permissions";
    }
    return "no";
}

struct ExecutionTrace
{
    ImplPath               impl_path              = ImplPath::Baseline;
    bool                   tensor_core_requested  = false;
    TensorCoreVerification tensor_core_verified   = TensorCoreVerification::No;
};

class BackendContext
{
  public:
    explicit BackendContext(Mode mode);
    ~BackendContext();

    BackendContext(const BackendContext&)            = delete;
    BackendContext& operator=(const BackendContext&) = delete;

    [[nodiscard]] Mode mode() const noexcept { return m_mode; }
    [[nodiscard]] const DeviceInfo& device_info() const noexcept { return m_device_info; }
    [[nodiscard]] bool is_supported() const noexcept;
    [[nodiscard]] std::string unsupported_reason() const;
    [[nodiscard]] const ExecutionTrace& trace() const noexcept { return m_trace; }
    void reset_trace() noexcept;
    void set_trace(ImplPath               impl_path,
                   bool                   tensor_core_requested,
                   TensorCoreVerification tensor_core_verified) noexcept;

    cublasStatus_t gemm_strided_batched(cublasOperation_t op_a,
                                        cublasOperation_t op_b,
                                        int               m,
                                        int               n,
                                        int               k,
                                        const double*     alpha,
                                        const double*     a,
                                        int               lda,
                                        long long         stride_a,
                                        const double*     b,
                                        int               ldb,
                                        long long         stride_b,
                                        const double*     beta,
                                        double*           c,
                                        int               ldc,
                                        long long         stride_c,
                                        int               batch_count);

    cublasStatus_t gemm_strided_batched(cublasOperation_t op_a,
                                        cublasOperation_t op_b,
                                        int               m,
                                        int               n,
                                        int               k,
                                        const float*      alpha,
                                        const float*      a,
                                        int               lda,
                                        long long         stride_a,
                                        const float*      b,
                                        int               ldb,
                                        long long         stride_b,
                                        const float*      beta,
                                        float*            c,
                                        int               ldc,
                                        long long         stride_c,
                                        int               batch_count);

    cublasStatus_t tc_blas_matmul_strided_batched(cublasOperation_t op_a,
                                                  cublasOperation_t op_b,
                                                  int               m,
                                                  int               n,
                                                  int               k,
                                                  const float*      alpha,
                                                  const float*      a,
                                                  int               lda,
                                                  long long         stride_a,
                                                  const float*      b,
                                                  int               ldb,
                                                  long long         stride_b,
                                                  const float*      beta,
                                                  const float*      c,
                                                  int               ldc,
                                                  long long         stride_c,
                                                  float*            d,
                                                  int               ldd,
                                                  long long         stride_d,
                                                  int               batch_count);

  private:
    Mode                 m_mode;
    DeviceInfo           m_device_info;
    cublasHandle_t       m_handle    = nullptr;
    cublasLtHandle_t     m_lt_handle = nullptr;
    DeviceBuffer<std::byte> m_lt_workspace;
    ExecutionTrace       m_trace;
};
}  // namespace uipc::tensor_core_lab
