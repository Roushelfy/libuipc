#include <sol/graph.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace sol
{
namespace
{
std::uint64_t edge_key(std::size_t a, std::size_t b)
{
    if(a > b)
        std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32) ^ static_cast<std::uint64_t>(b);
}

std::string trim(std::string value)
{
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char c) { return !is_space(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char c) { return !is_space(c); }).base(),
                value.end());
    return value;
}

std::vector<std::string> split_csv_line(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream        ss(line);
    std::string              item;
    while(std::getline(ss, item, ','))
        out.push_back(trim(item));
    return out;
}
} // namespace

void add_atom(AtomGraph& graph,
              std::size_t dof_count,
              std::string_view source_kind,
              std::size_t source_id)
{
    const std::size_t id = graph.atoms.size();
    if(source_id == static_cast<std::size_t>(-1))
        source_id = id;
    graph.atoms.push_back(Atom{id, source_id, dof_count, std::string(source_kind)});
}

void add_edge(AtomGraph& graph,
              std::size_t a,
              std::size_t b,
              double weight,
              std::string_view kind)
{
    if(a == b)
        return;
    if(a > b)
        std::swap(a, b);
    if(weight <= 0.0)
        throw std::invalid_argument("edge weight must be positive");

    const auto existing = std::find_if(graph.edges.begin(),
                                       graph.edges.end(),
                                       [a, b](const AtomEdge& edge)
                                       {
                                           return edge.a == a && edge.b == b;
                                       });
    if(existing != graph.edges.end())
    {
        existing->weight += weight;
        if(existing->kind != kind)
            existing->kind += "+" + std::string(kind);
        return;
    }
    graph.edges.push_back(AtomEdge{a, b, weight, std::string(kind)});
}

void validate_graph(const AtomGraph& graph)
{
    for(std::size_t i = 0; i < graph.atoms.size(); ++i)
    {
        if(graph.atoms[i].id != i)
            throw std::invalid_argument("atom ids must be dense and match their vector index");
        if(graph.atoms[i].dof_count == 0)
            throw std::invalid_argument("atom dof_count must be positive");
    }

    std::unordered_map<std::uint64_t, bool> seen;
    for(const auto& edge : graph.edges)
    {
        if(edge.a >= graph.atoms.size() || edge.b >= graph.atoms.size())
            throw std::invalid_argument("edge endpoint is out of range");
        if(edge.a == edge.b)
            throw std::invalid_argument("self edges are not allowed");
        if(edge.a > edge.b)
            throw std::invalid_argument("edges must be stored in canonical order");
        if(edge.weight <= 0.0)
            throw std::invalid_argument("edge weight must be positive");
        const auto key = edge_key(edge.a, edge.b);
        if(seen.contains(key))
            throw std::invalid_argument("duplicate edge found");
        seen[key] = true;
    }
}

AtomGraph relabel_graph(const AtomGraph& graph, const std::vector<std::size_t>& new_to_old)
{
    validate_graph(graph);
    if(new_to_old.size() != graph.atoms.size())
        throw std::invalid_argument("relabel permutation size must match atom count");

    std::vector<std::size_t> old_to_new(graph.atoms.size(), graph.atoms.size());
    for(std::size_t new_id = 0; new_id < new_to_old.size(); ++new_id)
    {
        const std::size_t old_id = new_to_old[new_id];
        if(old_id >= graph.atoms.size() || old_to_new[old_id] != graph.atoms.size())
            throw std::invalid_argument("relabel permutation must be complete and unique");
        old_to_new[old_id] = new_id;
    }

    AtomGraph out;
    out.name = graph.name + "_shuffled";
    out.atoms.resize(graph.atoms.size());
    for(std::size_t new_id = 0; new_id < new_to_old.size(); ++new_id)
    {
        Atom atom = graph.atoms[new_to_old[new_id]];
        atom.id   = new_id;
        out.atoms[new_id] = std::move(atom);
    }

    for(const auto& edge : graph.edges)
        add_edge(out, old_to_new[edge.a], old_to_new[edge.b], edge.weight, edge.kind);
    validate_graph(out);
    return out;
}

std::vector<std::vector<std::size_t>> adjacency(const AtomGraph& graph)
{
    validate_graph(graph);
    std::vector<std::vector<std::size_t>> out(graph.atoms.size());
    for(const auto& edge : graph.edges)
    {
        out[edge.a].push_back(edge.b);
        out[edge.b].push_back(edge.a);
    }
    for(auto& row : out)
        std::sort(row.begin(), row.end());
    return out;
}

AtomGraph graph_from_json(const nlohmann::json& json)
{
    AtomGraph graph;
    graph.name = json.value("name", "json_graph");

    if(json.contains("atoms"))
    {
        const auto& atoms = json.at("atoms");
        if(!atoms.is_array())
            throw std::invalid_argument("graph JSON field 'atoms' must be an array");
        for(std::size_t i = 0; i < atoms.size(); ++i)
        {
            const auto& atom = atoms.at(i);
            if(atom.is_object())
            {
                add_atom(graph,
                         atom.value<std::size_t>("dof_count", 3),
                         atom.value<std::string>("source_kind", "json"),
                         atom.value<std::size_t>("source_id", i));
            }
            else
            {
                add_atom(graph);
            }
        }
    }
    else
    {
        const std::size_t atom_count = json.value<std::size_t>("atom_count", 0);
        for(std::size_t i = 0; i < atom_count; ++i)
            add_atom(graph);
    }

    if(!json.contains("edges"))
        throw std::invalid_argument("graph JSON requires an 'edges' array");

    for(const auto& edge : json.at("edges"))
    {
        if(edge.is_array())
        {
            if(edge.size() < 2)
                throw std::invalid_argument("edge array entries need at least two endpoints");
            add_edge(graph,
                     edge.at(0).get<std::size_t>(),
                     edge.at(1).get<std::size_t>(),
                     edge.size() >= 3 ? edge.at(2).get<double>() : 1.0,
                     edge.size() >= 4 ? edge.at(3).get<std::string>() : "json");
        }
        else
        {
            add_edge(graph,
                     edge.at("a").get<std::size_t>(),
                     edge.at("b").get<std::size_t>(),
                     edge.value("weight", 1.0),
                     edge.value("kind", "json"));
        }
    }
    validate_graph(graph);
    return graph;
}

AtomGraph read_graph_json(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if(!in)
        throw std::runtime_error("failed to open graph JSON: " + path.string());
    nlohmann::json json;
    in >> json;
    return graph_from_json(json);
}

void read_extra_edges_csv(AtomGraph& graph, const std::filesystem::path& path)
{
    if(path.empty())
        return;

    std::ifstream in(path);
    if(!in)
        throw std::runtime_error("failed to open extra edge CSV: " + path.string());

    std::string line;
    bool        first = true;
    while(std::getline(in, line))
    {
        line = trim(line);
        if(line.empty() || line.starts_with('#'))
            continue;
        const auto fields = split_csv_line(line);
        if(first)
        {
            first = false;
            if(!fields.empty() && !fields[0].empty()
               && !std::isdigit(static_cast<unsigned char>(fields[0][0])))
                continue;
        }
        if(fields.size() < 2)
            throw std::invalid_argument("extra edge CSV rows must be a,b[,weight[,kind]]");
        const double weight = fields.size() >= 3 && !fields[2].empty() ? std::stod(fields[2]) : 1.0;
        const auto   kind   = fields.size() >= 4 && !fields[3].empty() ? fields[3] : "contact_hint";
        add_edge(graph, std::stoull(fields[0]), std::stoull(fields[1]), weight, kind);
    }
    validate_graph(graph);
}
} // namespace sol
