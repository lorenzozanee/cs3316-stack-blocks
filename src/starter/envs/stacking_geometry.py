"""Strict scene geometry for the stacking internal reward.

The helper reads only layout-manager bookkeeping: ``table_info`` written by
``LayoutManager.select_table`` and per-block instance metadata loaded from the
asset ``metadata.json``. It never walks a mesh (``RigidObject.get_bbox``) and
never launches Isaac Sim.

Coordinate frame: both ``table_info`` and ``get_label_pose``'s blocks use
scene-relative coordinates (the per-environment origin is not added). Table
bounds are therefore directly comparable with block positions.

Table bounds are axis-aligned. ``table_info`` stores
``size=[-sx/2, -sy/2, sx/2, sy/2]`` around ``pos`` and the table's top as
``height``; it does not store the table orientation. Every stack-blocks eval
layout (``Assets/Eval_Layout/RoboDojo/arx_x5/<seed>/*.json``) and the default
scene config use ``default_ori=[1, 0, 0, 0]``, so the stored bounds are the
world bounds. A rotated table would require a different representation.

Block extents come from ``geometry.oriented_bbox.extents`` in the instance
metadata. Those values are asset-local and unscaled; ``RigidObject`` applies
the layout record's ``scale`` to the USD prim. The helper therefore multiplies
the metadata extents by the instance's runtime ``scale``. All current
stack-blocks layouts use ``scale=[1, 1, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from starter.interfaces import BLOCK_LABELS


def stacking_geometry(task, env_index: int) -> dict:
    """Return strict scene geometry for one environment.

    Output (plain floats/lists, JSON-packable by ``remote._pack``)::

        {
            "table": {
                "top_z": float,
                "x_min": float, "x_max": float,
                "y_min": float, "y_max": float,
            },
            "block_extents": {
                "block_0": [x, y, z], ...,
            },
        }

    Missing or malformed layout-manager data raises ``ValueError``; there is no
    default geometry.
    """

    layout_manager = _layout_manager(task)
    table_info = _table_info(layout_manager, env_index)
    table = _table_bounds(table_info, env_index)

    get_metadata = getattr(layout_manager, "get_instance_metadata", None)
    if not callable(get_metadata):
        raise ValueError("stacking_geometry requires layout_manager.get_instance_metadata().")
    get_name = getattr(layout_manager, "get_instance_name", None)
    get_object = getattr(layout_manager, "get_scene_object", None)
    if not callable(get_name) or not callable(get_object):
        raise ValueError(
            "stacking_geometry requires layout_manager.get_instance_name() and "
            "get_scene_object() to read the instance scale."
        )

    block_extents = {}
    for label in BLOCK_LABELS:
        metadata = get_metadata(env_idx=env_index, label=label)
        extents = _metadata_extents(metadata, label, env_index)
        scale = _instance_scale(layout_manager, env_index, label)
        block_extents[label] = [
            extents[axis] * scale[axis] for axis in range(3)
        ]
    return {"table": table, "block_extents": block_extents}


def _layout_manager(task):
    scene_manager = getattr(task, "scene_manager", None)
    if scene_manager is None:
        raise ValueError("stacking_geometry requires task.scene_manager.")
    layout_manager = getattr(scene_manager, "layout_manager", None)
    if layout_manager is None:
        raise ValueError("stacking_geometry requires task.scene_manager.layout_manager.")
    return layout_manager


def _table_info(layout_manager, env_index: int) -> Mapping:
    table_infos = getattr(layout_manager, "table_info", None)
    if not isinstance(table_infos, (list, tuple)):
        raise ValueError("stacking_geometry requires layout_manager.table_info as a list.")
    if not isinstance(env_index, int) or isinstance(env_index, bool):
        raise ValueError(f"env_index must be an integer, got {env_index!r}.")
    if not 0 <= env_index < len(table_infos):
        raise ValueError(
            f"env_index {env_index} is outside layout_manager.table_info "
            f"(length {len(table_infos)})."
        )
    table_info = table_infos[env_index]
    if not isinstance(table_info, Mapping) or not table_info:
        raise ValueError(
            f"layout_manager.table_info[{env_index}] is empty; reset the scene first."
        )
    missing = [key for key in ("height", "pos", "size") if key not in table_info]
    if missing:
        raise ValueError(
            f"layout_manager.table_info[{env_index}] is missing {missing}."
        )
    return table_info


def _table_bounds(table_info: Mapping, env_index: int) -> dict:
    top_z = _finite_float(table_info["height"], f"table_info[{env_index}].height")
    pos = _finite_vector(table_info["pos"], 3, f"table_info[{env_index}].pos")
    size = _finite_vector(table_info["size"], 4, f"table_info[{env_index}].size")
    x_min, x_max = pos[0] + size[0], pos[0] + size[2]
    y_min, y_max = pos[1] + size[1], pos[1] + size[3]
    if not x_min < x_max:
        raise ValueError(
            f"table_info[{env_index}].size gives an empty x range [{x_min}, {x_max}]."
        )
    if not y_min < y_max:
        raise ValueError(
            f"table_info[{env_index}].size gives an empty y range [{y_min}, {y_max}]."
        )
    return {"top_z": top_z, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def _metadata_extents(metadata, label: str, env_index: int) -> tuple[float, float, float]:
    where = f"layout_manager.get_instance_metadata(env_idx={env_index}, label={label!r})"
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{where} returned no metadata.")
    geometry = metadata.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"{where} has no geometry metadata.")
    oriented = geometry.get("oriented_bbox")
    if not isinstance(oriented, Mapping) or "extents" not in oriented:
        raise ValueError(f"{where} has no geometry.oriented_bbox.extents.")
    extents = _finite_vector(oriented["extents"], 3, f"{where} geometry.oriented_bbox.extents")
    if any(value <= 0.0 for value in extents):
        raise ValueError(
            f"{where} geometry.oriented_bbox.extents must be positive, got {list(extents)}."
        )
    return extents


def _instance_scale(layout_manager, env_index: int, label: str) -> tuple[float, float, float]:
    inst_name = layout_manager.get_instance_name(env_index, label)
    if inst_name is None:
        raise ValueError(f"No scene instance is registered for label {label!r} in env {env_index}.")
    obj = layout_manager.get_scene_object(env_index, inst_name)
    if obj is None:
        raise ValueError(f"Scene object {inst_name!r} for label {label!r} is missing in env {env_index}.")
    scale = getattr(obj, "scale", None)
    if scale is None:
        raise ValueError(f"Scene object {inst_name!r} for label {label!r} has no scale.")
    values = _finite_vector(scale, 3, f"scene object {inst_name!r} scale")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"Scene object {inst_name!r} scale must be positive, got {list(values)}.")
    return values


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return number


def _finite_vector(value, width: int, name: str) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of {width} numbers, got {value!r}.")
    if len(value) != width:
        raise ValueError(f"{name} must have length {width}, got {len(value)}.")
    return tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value))
