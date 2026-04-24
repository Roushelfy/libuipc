#include <sol/presets.h>

#include <sol/graph.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>

namespace sol
{
namespace
{
std::vector<std::size_t> deterministic_shuffle(std::size_t n, std::uint32_t seed)
{
    std::vector<std::size_t> values(n);
    std::iota(values.begin(), values.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(values.begin(), values.end(), rng);
    return values;
}

void add_triangle(AtomGraph& graph, std::size_t a, std::size_t b, std::size_t c)
{
    add_edge(graph, a, b);
    add_edge(graph, b, c);
    add_edge(graph, a, c);
}

void add_tet(AtomGraph& graph, std::size_t a, std::size_t b, std::size_t c, std::size_t d)
{
    add_edge(graph, a, b);
    add_edge(graph, a, c);
    add_edge(graph, a, d);
    add_edge(graph, b, c);
    add_edge(graph, b, d);
    add_edge(graph, c, d);
}
} // namespace

AtomGraph make_rod(std::size_t count)
{
    AtomGraph graph;
    graph.name = "rod";
    for(std::size_t i = 0; i < count; ++i)
        add_atom(graph, 3, "rod_vertex", i);
    for(std::size_t i = 1; i < count; ++i)
        add_edge(graph, i - 1, i);
    validate_graph(graph);
    return graph;
}

AtomGraph make_cloth_grid(std::size_t width, std::size_t height)
{
    if(width < 2 || height < 2)
        throw std::invalid_argument("cloth_grid requires width and height >= 2");

    AtomGraph graph;
    graph.name = "cloth_grid";
    for(std::size_t i = 0; i < width * height; ++i)
        add_atom(graph, 3, "cloth_vertex", i);

    auto id = [width](std::size_t x, std::size_t y) { return y * width + x; };
    for(std::size_t y = 0; y + 1 < height; ++y)
    {
        for(std::size_t x = 0; x + 1 < width; ++x)
        {
            add_triangle(graph, id(x, y), id(x + 1, y), id(x + 1, y + 1));
            add_triangle(graph, id(x, y), id(x + 1, y + 1), id(x, y + 1));
        }
    }
    validate_graph(graph);
    return graph;
}

AtomGraph make_tet_block(std::size_t nx, std::size_t ny, std::size_t nz)
{
    if(nx == 0 || ny == 0 || nz == 0)
        throw std::invalid_argument("tet_block dimensions must be positive");

    AtomGraph graph;
    graph.name = "tet_block";
    const std::size_t sx = nx + 1;
    const std::size_t sy = ny + 1;
    auto id = [sx, sy](std::size_t x, std::size_t y, std::size_t z)
    {
        return z * sx * sy + y * sx + x;
    };

    for(std::size_t i = 0; i < sx * sy * (nz + 1); ++i)
        add_atom(graph, 3, "tet_vertex", i);

    for(std::size_t z = 0; z < nz; ++z)
    {
        for(std::size_t y = 0; y < ny; ++y)
        {
            for(std::size_t x = 0; x < nx; ++x)
            {
                const std::size_t v000 = id(x, y, z);
                const std::size_t v100 = id(x + 1, y, z);
                const std::size_t v010 = id(x, y + 1, z);
                const std::size_t v110 = id(x + 1, y + 1, z);
                const std::size_t v001 = id(x, y, z + 1);
                const std::size_t v101 = id(x + 1, y, z + 1);
                const std::size_t v011 = id(x, y + 1, z + 1);
                const std::size_t v111 = id(x + 1, y + 1, z + 1);

                add_tet(graph, v000, v100, v010, v001);
                add_tet(graph, v100, v110, v010, v111);
                add_tet(graph, v100, v010, v001, v111);
                add_tet(graph, v100, v001, v101, v111);
                add_tet(graph, v010, v001, v011, v111);
            }
        }
    }
    validate_graph(graph);
    return graph;
}

AtomGraph make_shuffled_cloth_grid()
{
    return relabel_graph(make_cloth_grid(), deterministic_shuffle(16 * 12, 74017));
}

AtomGraph make_shuffled_tet_block()
{
    const auto graph = make_tet_block();
    return relabel_graph(graph, deterministic_shuffle(graph.atoms.size(), 74019));
}

AtomGraph make_preset(std::string_view name)
{
    if(name == "rod")
        return make_rod();
    if(name == "cloth_grid")
        return make_cloth_grid();
    if(name == "tet_block")
        return make_tet_block();
    if(name == "shuffled_cloth_grid")
        return make_shuffled_cloth_grid();
    if(name == "shuffled_tet_block")
        return make_shuffled_tet_block();
    throw std::invalid_argument(
        "unsupported preset; expected rod, cloth_grid, tet_block, shuffled_cloth_grid, or shuffled_tet_block");
}
} // namespace sol
