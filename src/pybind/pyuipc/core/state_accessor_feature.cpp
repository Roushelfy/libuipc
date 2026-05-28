#include <pyuipc/core/state_accessor_feature.h>
#include <uipc/core/finite_element_state_accessor_feature.h>
#include <uipc/core/affine_body_state_accessor_feature.h>
#include <uipc/core/rcc_adhesion_state_accessor_feature.h>

namespace pyuipc::core
{
using namespace uipc::core;

PyStateAccessorFeature::PyStateAccessorFeature(py::module& m)
{
    auto class_FiniteElementStateAccessorFeature =
        py::class_<FiniteElementStateAccessorFeature, IFeature, S<FiniteElementStateAccessorFeature>>(
            m,
            "FiniteElementStateAccessorFeature",
            R"(Feature for accessing finite element simulation state (vertex positions, velocities, etc.).)");

    class_FiniteElementStateAccessorFeature.def("vertex_count",
                                                &FiniteElementStateAccessorFeature::vertex_count,
                                                R"(Get the number of vertices.
Returns:
    int: Number of vertices.)");

    class_FiniteElementStateAccessorFeature.def("create_geometry",
                                                &FiniteElementStateAccessorFeature::create_geometry,
                                                py::arg("vertex_offset") = 0,
                                                py::arg("vertex_count") = ~0ull,
                                                R"(Create geometry from state data.
Args:
    vertex_offset: Starting vertex index (default: 0).
    vertex_count: Number of vertices to include (default: all).
Returns:
    Geometry: Geometry created from state data.)");

    class_FiniteElementStateAccessorFeature.def("copy_from",
                                                &FiniteElementStateAccessorFeature::copy_from,
                                                py::arg("state_geo"),
                                                R"(Copy state from geometry.
Args:
    state_geo: Geometry containing state data to copy from.)");

    class_FiniteElementStateAccessorFeature.def("copy_to",
                                                &FiniteElementStateAccessorFeature::copy_to,
                                                py::arg("state_geo"),
                                                R"(Copy state to geometry.
Args:
    state_geo: Geometry to copy state data to.)");

    class_FiniteElementStateAccessorFeature.def("copy_position_to",
                                                &FiniteElementStateAccessorFeature::copy_position_to,
                                                py::arg("buffer_view"),
                                                py::arg("vertex_offset") = 0,
                                                py::arg("vertex_count")  = ~0ull,
                                                R"(Copy position data (Vector3) for the specified vertex range into an externally-owned buffer.
Args:
    buffer_view: Destination buffer view to copy position data into.
    vertex_offset: Starting vertex index (default: 0).
    vertex_count: Number of vertices to include (default: all).)");

    class_FiniteElementStateAccessorFeature.def("copy_velocity_to",
                                                &FiniteElementStateAccessorFeature::copy_velocity_to,
                                                py::arg("buffer_view"),
                                                py::arg("vertex_offset") = 0,
                                                py::arg("vertex_count")  = ~0ull,
                                                R"(Copy velocity data (Vector3) for the specified vertex range into an externally-owned buffer.
Args:
    buffer_view: Destination buffer view to copy velocity data into.
    vertex_offset: Starting vertex index (default: 0).
    vertex_count: Number of vertices to include (default: all).)");


    class_FiniteElementStateAccessorFeature.attr("FeatureName") =
        FiniteElementStateAccessorFeature::FeatureName;


    auto class_AffineBodyStateAccessorFeature =
        py::class_<AffineBodyStateAccessorFeature, IFeature, S<AffineBodyStateAccessorFeature>>(
            m,
            "AffineBodyStateAccessorFeature",
            R"(Feature for accessing affine body simulation state (body transforms, velocities, etc.).)");

    class_AffineBodyStateAccessorFeature.def("body_count",
                                             &AffineBodyStateAccessorFeature::body_count,
                                             R"(Get the number of affine bodies.
Returns:
    int: Number of affine bodies.)");

    class_AffineBodyStateAccessorFeature.def("create_geometry",
                                             &AffineBodyStateAccessorFeature::create_geometry,
                                             py::arg("body_offset") = 0,
                                             py::arg("body_count")  = ~0ull,
                                             R"(Create geometry from state data.
Args:
    body_offset: Starting body index (default: 0).
    body_count: Number of bodies to include (default: all).
Returns:
    Geometry: Geometry created from state data.)");

    class_AffineBodyStateAccessorFeature.def("copy_from",
                                             &AffineBodyStateAccessorFeature::copy_from,
                                             py::arg("state_geo"),
                                             R"(Copy state from geometry.
Args:
    state_geo: Geometry containing state data to copy from.)");

    class_AffineBodyStateAccessorFeature.def("copy_to",
                                             &AffineBodyStateAccessorFeature::copy_to,
                                             py::arg("state_geo"),
                                             R"(Copy state to geometry.
Args:
    state_geo: Geometry to copy state data to.)");

    class_AffineBodyStateAccessorFeature.def("copy_transform_to",
                                             &AffineBodyStateAccessorFeature::copy_transform_to,
                                             py::arg("buffer_view"),
                                             py::arg("body_offset") = 0,
                                             py::arg("body_count")  = ~0ull,
                                             R"(Copy transform data (Matrix4x4) for the specified body range into an externally-owned buffer.
Args:
    buffer_view: Destination buffer view to copy transform data into.
    body_offset: Starting body index (default: 0).
    body_count: Number of bodies to include (default: all).)");

    class_AffineBodyStateAccessorFeature.def("copy_velocity_to",
                                             &AffineBodyStateAccessorFeature::copy_velocity_to,
                                             py::arg("buffer_view"),
                                             py::arg("body_offset") = 0,
                                             py::arg("body_count")  = ~0ull,
                                             R"(Copy velocity data (Matrix4x4) for the specified body range into an externally-owned buffer.
Args:
    buffer_view: Destination buffer view to copy velocity data into.
    body_offset: Starting body index (default: 0).
    body_count: Number of bodies to include (default: all).)");


    class_AffineBodyStateAccessorFeature.attr("FeatureName") =
        AffineBodyStateAccessorFeature::FeatureName;


    auto class_RCCAdhesionStateAccessorFeature =
        py::class_<RCCAdhesionStateAccessorFeature, IFeature, S<RCCAdhesionStateAccessorFeature>>(
            m,
            "RCCAdhesionStateAccessorFeature",
            R"(Feature for round-tripping RCC adhesion per-pair β across processes.

The CUDA backend persists β across `world.advance()` steps via sorted
u64 vertex-tuple hash keys, but that persistence is in-memory only.
This feature exposes the same (keys, β) snapshot to Python so that an
application can save it (e.g. into a .npz alongside the geometry) and
restore it on a follow-up load — letting wound-state β survive a wind
→ save → reload cycle.

Only the PT (point-triangle) pair list is persisted (EE/PE/PP β is
disabled in v2).)");

    class_RCCAdhesionStateAccessorFeature.def(
        "pt_pair_count",
        &RCCAdhesionStateAccessorFeature::pt_pair_count,
        R"(Number of PT contact pairs whose β is currently held in the reporter's
prev-state snapshot.)");

    class_RCCAdhesionStateAccessorFeature.def(
        "dump_pt_state",
        [](const RCCAdhesionStateAccessorFeature& self)
        {
            uipc::vector<uipc::U64>   keys;
            uipc::vector<uipc::Float> betas;
            self.dump_pt_state(keys, betas);

            py::array_t<uint64_t> py_keys(static_cast<py::ssize_t>(keys.size()));
            py::array_t<double>   py_betas(static_cast<py::ssize_t>(betas.size()));
            if(!keys.empty())
                std::memcpy(py_keys.mutable_data(),  keys.data(),  keys.size()  * sizeof(uint64_t));
            if(!betas.empty())
                std::memcpy(py_betas.mutable_data(), betas.data(), betas.size() * sizeof(double));
            return py::make_tuple(py_keys, py_betas);
        },
        R"(Pull the prev-state (keys, β) snapshot from the device.

Returns:
    (keys, betas): a tuple of two numpy arrays — `keys` is uint64, `betas` is
    float64, both length pt_pair_count(). Index-aligned and sorted by key.)");

    class_RCCAdhesionStateAccessorFeature.def(
        "load_pt_state",
        [](const RCCAdhesionStateAccessorFeature& self,
           py::array_t<uint64_t, py::array::c_style | py::array::forcecast> keys,
           py::array_t<double,   py::array::c_style | py::array::forcecast> betas)
        {
            if(keys.size() != betas.size())
                throw std::invalid_argument(
                    "load_pt_state: keys.size() must equal betas.size()");
            // Reinterpret double → uipc::Float (== double on this build).
            static_assert(sizeof(uipc::Float) == sizeof(double),
                          "uipc::Float must be double for the pybind cast");
            uipc::span<const uipc::U64> key_span(
                reinterpret_cast<const uipc::U64*>(keys.data()),
                static_cast<std::size_t>(keys.size()));
            uipc::span<const uipc::Float> beta_span(
                reinterpret_cast<const uipc::Float*>(betas.data()),
                static_cast<std::size_t>(betas.size()));
            self.load_pt_state(key_span, beta_span);
        },
        py::arg("keys"),
        py::arg("betas"),
        R"(Push (keys, β) into the reporter as the prev-state snapshot.

Must be called after `world.init(scene)` and before the first
`world.advance()`. `keys` and `betas` must be 1-D numpy arrays of equal
length (uint64 and float64 respectively).)");

    class_RCCAdhesionStateAccessorFeature.attr("FeatureName") =
        RCCAdhesionStateAccessorFeature::FeatureName;
}
}  // namespace pyuipc::core
