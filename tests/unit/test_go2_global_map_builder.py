from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GLOBAL_MAP_MODULE = _load_module(
    "go2_global_map_test_module",
    _REPO_ROOT / "drivers" / "robots" / "go2" / "global_map.py",
)
_SETTINGS_MODULE = _load_module(
    "go2_settings_test_module",
    _REPO_ROOT / "drivers" / "robots" / "go2" / "settings.py",
)

Go2SparseGlobalMapBuilder = _GLOBAL_MAP_MODULE.Go2SparseGlobalMapBuilder
Go2MapSynthesisConfig = _SETTINGS_MODULE.Go2MapSynthesisConfig


def _grid_index(*, width: int, resolution_m: float, origin_x: float, origin_y: float, x: float, y: float) -> int:
    col = int((x - origin_x) / resolution_m)
    row = int((y - origin_y) / resolution_m)
    return row * width + col


def test_sparse_global_map_builder_accumulates_points_from_multiple_robot_positions() -> None:
    builder = Go2SparseGlobalMapBuilder(
        Go2MapSynthesisConfig(
            global_map_enabled=True,
            global_map_resolution_m=0.5,
            global_map_padding_cells=0,
            global_map_max_width=40,
            global_map_max_height=40,
            global_map_inflation_radius_m=0.0,
        )
    )

    first_scan = np.asarray([[1.0, 0.0, 0.1], [1.5, 0.0, 0.1]], dtype=np.float32)
    second_scan = np.asarray([[3.0, 0.0, 0.1], [3.5, 0.0, 0.1]], dtype=np.float32)

    assert builder.ingest_scan(world_points=first_scan, sensor_x=0.0, sensor_y=0.0, source_label="direct_dds_point_cloud")
    assert builder.ingest_scan(world_points=second_scan, sensor_x=2.0, sensor_y=0.0, source_label="direct_dds_point_cloud")

    build_result = builder.build()

    assert build_result is not None
    assert build_result.known_cell_count >= 4
    assert build_result.source_label == "direct_dds_point_cloud"

    occupancy = build_result.occupancy_data.flatten().tolist()
    first_obstacle_index = _grid_index(
        width=build_result.width,
        resolution_m=build_result.resolution_m,
        origin_x=build_result.origin_x,
        origin_y=build_result.origin_y,
        x=1.0,
        y=0.0,
    )
    second_obstacle_index = _grid_index(
        width=build_result.width,
        resolution_m=build_result.resolution_m,
        origin_x=build_result.origin_x,
        origin_y=build_result.origin_y,
        x=3.0,
        y=0.0,
    )
    free_cell_index = _grid_index(
        width=build_result.width,
        resolution_m=build_result.resolution_m,
        origin_x=build_result.origin_x,
        origin_y=build_result.origin_y,
        x=2.0,
        y=0.0,
    )

    assert occupancy[first_obstacle_index] == 100
    assert occupancy[second_obstacle_index] == 100
    assert occupancy[free_cell_index] == 0


def test_sparse_global_map_builder_can_restore_persisted_state() -> None:
    builder = Go2SparseGlobalMapBuilder(
        Go2MapSynthesisConfig(
            global_map_enabled=True,
            global_map_resolution_m=0.5,
            global_map_padding_cells=0,
            global_map_max_width=40,
            global_map_max_height=40,
            global_map_inflation_radius_m=0.0,
        )
    )
    scan = np.asarray([[1.0, 0.0, 0.1], [2.0, 0.0, 0.1]], dtype=np.float32)
    assert builder.ingest_scan(world_points=scan, sensor_x=0.0, sensor_y=0.0, source_label="direct_dds_point_cloud")

    exported_state = builder.export_state()
    restored_builder = Go2SparseGlobalMapBuilder(builder.config)

    assert restored_builder.restore_state(exported_state) is True

    original = builder.build()
    restored = restored_builder.build()

    assert original is not None
    assert restored is not None
    assert restored.known_cell_count == original.known_cell_count
    assert restored.source_label == original.source_label
    assert restored.occupancy_data.tolist() == original.occupancy_data.tolist()
