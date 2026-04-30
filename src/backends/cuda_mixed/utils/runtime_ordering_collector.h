#pragma once

#include <uipc/common/type_define.h>
#include <muda/buffer/buffer_view.h>

namespace uipc::backend::cuda_mixed
{
struct RuntimeOrderingEdge
{
    IndexT atom_a = -1;
    IndexT atom_b = -1;
    double abs_weight = 0.0;
};

struct RuntimeOrderingCollector
{
    muda::BufferView<RuntimeOrderingEdge> edges;
    muda::BufferView<IndexT>              cursor;
    muda::CBufferView<IndexT>             old_dof_to_atom;
    bool                                  enabled = false;
    bool                                  graph_only = false;
    bool                                  topology_only = false;

    MUDA_GENERIC bool valid() const noexcept
    {
        return enabled && edges.data() != nullptr && cursor.data() != nullptr
               && cursor.size() >= 2 && old_dof_to_atom.data() != nullptr;
    }
};
}  // namespace uipc::backend::cuda_mixed
