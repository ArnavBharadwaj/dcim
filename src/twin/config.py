"""Loading twin configuration from YAML into validated dataclasses.

Ground rule from the brief: every experiment is a config file plus a seed, and no
hyperparameter is hardcoded in a script. This module is the single door between YAML
and the twin, so an unrecognised key is an error rather than a silently ignored typo
-- a misspelled `escape_base` that falls back to the default would be exactly the
kind of silent bug that quietly invalidates a results table.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

from .geometry import HallGeometry, build_hall
from .power import PowerParams
from .recirculation import RecircParams
from .thermal import ThermalParams, ThermalTwin

HALL_KEYS = {"name", "rows", "racks_per_row", "orientation_pattern",
             "rack_width_m", "rack_depth_m", "aisle_width_m", "crac_units",
             "recirculation", "limits"}


@dataclasses.dataclass(frozen=True)
class Limits:
    """Operating envelope. Thresholds come from ASHRAE 2021 class A1."""

    inlet_recommended_max_c: float = 27.0
    inlet_allowable_max_c: float = 32.0
    supply_temp_min_c: float = 16.0
    supply_temp_max_c: float = 25.0
    crac_fan_frac_min: float = 0.40
    crac_fan_frac_max: float = 1.0


@dataclasses.dataclass(frozen=True)
class TwinConfig:
    recirculation: RecircParams
    power: PowerParams
    thermal: ThermalParams
    limits: Limits


def _strict(cls, mapping: dict[str, Any], where: str):
    """Build a dataclass, rejecting unknown keys instead of dropping them."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(mapping) - known
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(unknown)}; expected some of {sorted(known)}")
    return cls(**mapping)


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_twin_config(path: str | pathlib.Path,
                     recirc_override: dict[str, Any] | None = None,
                     limits_override: dict[str, Any] | None = None) -> TwinConfig:
    """Read configs/twin/*.yaml.

    A hall may override its own `recirculation` and `limits` without forking the whole
    parameter file. Limits are hall-specific for a physical reason: an uncontained hall
    recirculates over a much shorter path, so it runs far hotter at the same setpoints
    and is actually operated with colder supply air and higher fan speeds. Holding all
    three halls to one envelope would either put hall_c permanently in violation or
    leave hall_a with no decisions to make."""
    raw = load_yaml(path)
    unknown = set(raw) - {"recirculation", "power", "thermal", "limits"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}")

    recirc = dict(raw.get("recirculation", {}))
    if recirc_override:
        bad = set(recirc_override) - {f.name for f in dataclasses.fields(RecircParams)}
        if bad:
            raise ValueError(f"hall recirculation override: unknown key(s) {sorted(bad)}")
        recirc.update(recirc_override)

    limits = dict(raw.get("limits", {}))
    if limits_override:
        bad = set(limits_override) - {f.name for f in dataclasses.fields(Limits)}
        if bad:
            raise ValueError(f"hall limits override: unknown key(s) {sorted(bad)}")
        limits.update(limits_override)

    return TwinConfig(
        recirculation=_strict(RecircParams, recirc, f"{path}:recirculation"),
        power=_strict(PowerParams, dict(raw.get("power", {})), f"{path}:power"),
        thermal=_strict(ThermalParams, dict(raw.get("thermal", {})), f"{path}:thermal"),
        limits=_strict(Limits, limits, f"{path}:limits"),
    )


def load_hall_config(path: str | pathlib.Path) -> dict[str, Any]:
    raw = load_yaml(path)
    unknown = set(raw) - HALL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; "
                         f"expected some of {sorted(HALL_KEYS)}")
    for required in ("rows", "racks_per_row", "rack_width_m", "rack_depth_m",
                     "aisle_width_m", "crac_units"):
        if required not in raw:
            raise ValueError(f"{path}: missing required key {required!r}")
    return raw


def build(hall_path: str | pathlib.Path,
          twin_path: str | pathlib.Path) -> tuple[ThermalTwin, HallGeometry, TwinConfig]:
    """Load a hall and a parameter set and return a ready twin."""
    hall_raw = load_hall_config(hall_path)
    cfg = load_twin_config(twin_path, hall_raw.get("recirculation"),
                           hall_raw.get("limits"))
    geom = build_hall(hall_raw)
    twin = ThermalTwin(geom, recirc=cfg.recirculation, power=cfg.power,
                       thermal=cfg.thermal)
    return twin, geom, cfg
