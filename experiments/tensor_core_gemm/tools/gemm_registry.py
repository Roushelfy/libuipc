from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    tag: str
    group: str
    m: int
    n: int
    k: int


ALL_SHAPES = [
    Shape("3x3x3", "uipc_common_square", 3, 3, 3),
    Shape("6x6x6", "uipc_common_square", 6, 6, 6),
    Shape("9x9x9", "uipc_common_square", 9, 9, 9),
    Shape("12x12x12", "uipc_common_square", 12, 12, 12),
    Shape("24x24x24", "uipc_common_square", 24, 24, 24),
    Shape("48x48x48", "uipc_common_square", 48, 48, 48),
    Shape("3x3x12", "uipc_common_rect", 3, 3, 12),
    Shape("12x12x3", "uipc_common_rect", 12, 12, 3),
    Shape("9x12x9", "uipc_common_rect", 9, 12, 9),
    Shape("12x12x9", "uipc_common_rect", 12, 12, 9),
    Shape("3x24x3", "uipc_common_rect", 3, 24, 3),
    Shape("24x24x3", "uipc_common_rect", 24, 24, 3),
    Shape("9x24x9", "uipc_common_rect", 9, 24, 9),
    Shape("24x24x9", "uipc_common_rect", 24, 24, 9),
    Shape("16x16x16", "friendly_square", 16, 16, 16),
    Shape("32x32x32", "friendly_square", 32, 32, 32),
    Shape("64x64x64", "friendly_square", 64, 64, 64),
    Shape("96x96x96", "friendly_square", 96, 96, 96),
    Shape("128x128x128", "friendly_square", 128, 128, 128),
    Shape("5x5x5", "awkward_square", 5, 5, 5),
    Shape("7x7x7", "awkward_square", 7, 7, 7),
    Shape("15x15x15", "awkward_square", 15, 15, 15),
    Shape("20x20x20", "awkward_square", 20, 20, 20),
    Shape("40x40x40", "awkward_square", 40, 40, 40),
    Shape("16x32x16", "friendly_rect", 16, 32, 16),
    Shape("32x32x16", "friendly_rect", 32, 32, 16),
    Shape("32x64x32", "friendly_rect", 32, 64, 32),
    Shape("64x128x64", "friendly_rect", 64, 128, 64),
    Shape("5x3x7", "awkward_rect", 5, 3, 7),
    Shape("7x5x15", "awkward_rect", 7, 5, 15),
    Shape("15x7x20", "awkward_rect", 15, 7, 20),
    Shape("20x15x40", "awkward_rect", 20, 15, 40),
]

REPRESENTATIVE_PROFILE_SHAPES = {
    "12x12x12": ("Padded", 12, 12, 12),
    "16x16x16_raw": ("Raw", 16, 16, 16),
    "16x16x16_padded": ("Padded", 16, 16, 16),
    "24x24x24": ("Padded", 24, 24, 24),
    "48x48x48": ("Padded", 48, 48, 48),
    "16x32x16": ("Padded", 16, 32, 16),
    "64x128x64": ("Padded", 64, 128, 64),
    "15x15x15": ("Raw", 15, 15, 15),
}


def round_up_to_multiple_of_16(value: int) -> int:
    return ((value + 15) // 16) * 16


def logical_flops(shape: Shape) -> int:
    return 2 * shape.m * shape.n * shape.k


def smoke_batches(shape: Shape) -> list[int]:
    flops = logical_flops(shape)
    if flops <= 4096:
        return [16384]
    if flops <= 65536:
        return [4096]
    return [1024]


def full_batches(shape: Shape) -> list[int]:
    flops = logical_flops(shape)
    if flops <= 4096:
        return [16384, 65536, 262144]
    if flops <= 65536:
        return [4096, 16384, 65536]
    return [1024, 4096, 16384]


def layout_variant_name(layout: str) -> str:
    return "raw" if layout == "Raw" else "padded"


def physical_dims(layout: str, m: int, n: int, k: int) -> tuple[int, int, int]:
    if layout == "Raw":
        return m, n, k
    return (
        round_up_to_multiple_of_16(m),
        round_up_to_multiple_of_16(n),
        round_up_to_multiple_of_16(k),
    )


def benchmark_name(layout: str, mode: str, shape: Shape, batch: int) -> str:
    return f"BM_Gemm{layout}{mode}/{shape.m}/{shape.n}/{shape.k}/{batch}/manual_time"


def find_shape(m: int, n: int, k: int) -> Shape:
    for shape in ALL_SHAPES:
        if (shape.m, shape.n, shape.k) == (m, n, k):
            return shape
    raise ValueError(f"unknown shape ({m}, {n}, {k})")
