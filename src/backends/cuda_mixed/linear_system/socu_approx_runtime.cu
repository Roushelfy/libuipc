#include <linear_system/socu_approx_runtime.h>

#include <uipc/common/json.h>

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <filesystem>
#include <fstream>
#include <initializer_list>

#ifndef UIPC_WITH_SOCU_NATIVE
#define UIPC_WITH_SOCU_NATIVE 0
#endif

#if UIPC_WITH_SOCU_NATIVE
#ifndef SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH
#define SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH ""
#endif
#endif

namespace uipc::backend::cuda_mixed
{
namespace fs = std::filesystem;

#if UIPC_WITH_SOCU_NATIVE
fs::path default_mathdx_manifest_path()
{
    return fs::path{SOCU_NATIVE_DEFAULT_MATHDX_MANIFEST_PATH};
}

std::string to_report_string(socu_native::SolverBackend backend)
{
    switch(backend)
    {
        case socu_native::SolverBackend::NativeProof:
            return "native_proof";
        case socu_native::SolverBackend::NativePerf:
            return "native_perf";
        case socu_native::SolverBackend::CpuEigen:
            return "cpu_eigen";
    }
    return "unknown";
}

std::string to_report_string(socu_native::PerfBackend backend)
{
    switch(backend)
    {
        case socu_native::PerfBackend::Auto:
            return "auto";
        case socu_native::PerfBackend::Native:
            return "native";
        case socu_native::PerfBackend::CublasLt:
            return "cublaslt";
        case socu_native::PerfBackend::MathDx:
            return "mathdx";
    }
    return "unknown";
}

std::string to_report_string(socu_native::MathMode mode)
{
    switch(mode)
    {
        case socu_native::MathMode::Auto:
            return "auto";
        case socu_native::MathMode::Strict:
            return "strict";
        case socu_native::MathMode::TF32:
            return "tf32";
    }
    return "unknown";
}

std::string to_report_string(socu_native::GraphMode mode)
{
    switch(mode)
    {
        case socu_native::GraphMode::Off:
            return "off";
        case socu_native::GraphMode::On:
            return "on";
        case socu_native::GraphMode::Auto:
            return "auto";
    }
    return "unknown";
}

std::string mathdx_bundle_key(std::string_view prefix,
                              std::string_view dtype,
                              SizeT            block_size)
{
    return fmt::format("{}_{}_n{}", prefix, dtype, block_size);
}

fs::path manifest_relative_path(const fs::path& manifest_path, const std::string& path)
{
    fs::path p{path};
    if(p.is_relative())
        p = manifest_path.parent_path() / p;
    return p;
}

bool artifact_ref_ready(const Json&     artifact,
                        const fs::path& manifest_path,
                        std::string&    detail)
{
    if(!artifact.is_object())
    {
        detail = "MathDx artifact entry is not an object";
        return false;
    }
    const auto symbol = artifact.value("symbol", std::string{});
    const auto lto    = artifact.value("lto", std::string{});
    if(symbol.empty() || lto.empty())
    {
        detail = "MathDx artifact is missing symbol or lto";
        return false;
    }
    const auto lto_path = manifest_relative_path(manifest_path, lto);
    if(!fs::is_regular_file(lto_path))
    {
        detail = fmt::format("MathDx artifact lto '{}' is missing", lto_path.string());
        return false;
    }
    const auto fatbin = artifact.value("fatbin", std::string{});
    if(!fatbin.empty())
    {
        const auto fatbin_path = manifest_relative_path(manifest_path, fatbin);
        if(!fs::is_regular_file(fatbin_path))
        {
            detail =
                fmt::format("MathDx artifact fatbin '{}' is missing", fatbin_path.string());
            return false;
        }
    }
    return true;
}

template <typename Scalar>
bool validate_mathdx_manifest(const fs::path& manifest_path,
                              SizeT          block_size,
                              SocuApproxGateReport& gate,
                              std::string& detail)
{
    gate.mathdx_manifest_path = manifest_path.string();
    gate.mathdx_runtime_cache_dir = (manifest_path.parent_path() / "runtime").string();

    std::ifstream ifs{manifest_path};
    if(!ifs)
    {
        detail = fmt::format("MathDx manifest '{}' cannot be opened",
                             manifest_path.string());
        return false;
    }
    Json manifest = Json::parse(ifs, nullptr, false);
    if(manifest.is_discarded() || !manifest.is_object())
    {
        detail = "MathDx manifest is not valid JSON";
        return false;
    }
    if(!manifest.value("mathdx_enabled", false))
    {
        detail = "MathDx manifest reports mathdx_enabled=false";
        return false;
    }

    int device = 0;
    SOCU_NATIVE_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    SOCU_NATIVE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    const int device_arch = prop.major * 10 + prop.minor;
    const int manifest_arch = manifest.value("arch", 0);
    if(manifest_arch != device_arch)
    {
        detail = fmt::format("MathDx manifest arch mismatch: manifest arch={}, device arch={}",
                             manifest_arch,
                             device_arch);
        return false;
    }

    const std::string dtype = socu_dtype_name<Scalar>();
    const Json* runtime_backend = nullptr;
    auto runtime_it = manifest.find("runtime_backend");
    if(runtime_it != manifest.end() && runtime_it->is_object())
        runtime_backend = &*runtime_it;
    if(!runtime_backend)
    {
        detail = "MathDx manifest is missing runtime_backend";
        return false;
    }

    auto validate_bundle = [&](const std::string& key,
                               std::initializer_list<const char*> required_keys) -> bool
    {
        auto bundle_it = runtime_backend->find(key);
        if(bundle_it == runtime_backend->end() || !bundle_it->is_object())
        {
            detail = fmt::format("MathDx manifest is missing runtime bundle '{}'", key);
            return false;
        }
        const auto cusolverdx_fatbin =
            bundle_it->value("cusolverdx_fatbin", std::string{});
        if(!cusolverdx_fatbin.empty()
           && !fs::is_regular_file(manifest_relative_path(manifest_path,
                                                          cusolverdx_fatbin)))
        {
            detail = fmt::format("MathDx runtime bundle '{}' has a missing cusolverdx_fatbin",
                                 key);
            return false;
        }
        for(const char* required_key : required_keys)
        {
            auto artifact_it = bundle_it->find(required_key);
            if(artifact_it == bundle_it->end())
            {
                detail = fmt::format("MathDx runtime bundle '{}' is missing '{}'",
                                     key,
                                     required_key);
                return false;
            }
            if(!artifact_ref_ready(*artifact_it, manifest_path, detail))
                return false;
        }
        return true;
    };

    const std::string factor_key = mathdx_bundle_key("factor", dtype, block_size);
    if(!validate_bundle(factor_key, {"potrf", "trsm_llnn"}))
        return false;

    auto factor_bundle = runtime_backend->find(factor_key);
    auto rltn_it       = factor_bundle->find("trsm_rltn_candidates");
    if(rltn_it == factor_bundle->end() || !rltn_it->is_array() || rltn_it->empty())
    {
        detail =
            fmt::format("MathDx runtime bundle '{}' is missing trsm_rltn_candidates",
                        factor_key);
        return false;
    }
    for(const Json& candidate : *rltn_it)
    {
        if(!artifact_ref_ready(candidate, manifest_path, detail))
            return false;
    }

    const std::string solve_key = mathdx_bundle_key("solve", dtype, block_size);
    if(!validate_bundle(solve_key,
                        {"potrs", "trsm_lower_rhs_1", "trsm_upper_rhs_1"}))
    {
        return false;
    }

    if(!fs::is_directory(manifest_path.parent_path()))
    {
        detail = "MathDx manifest parent directory is unavailable";
        return false;
    }

    gate.mathdx_manifest_ok = true;
    gate.mathdx_artifacts_ok = true;
    return true;
}


template bool validate_mathdx_manifest<float>(const fs::path&,
                                              SizeT,
                                              SocuApproxGateReport&,
                                              std::string&);
template bool validate_mathdx_manifest<double>(const fs::path&,
                                               SizeT,
                                               SocuApproxGateReport&,
                                               std::string&);
#endif
}  // namespace uipc::backend::cuda_mixed
