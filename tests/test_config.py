"""Tests for config loading.

A misspelled key that silently falls back to a default is the archetypal silent bug:
the run completes, the CSV row looks fine, and the number is wrong. The loader is
strict, and these tests hold it strict.
"""

import pytest

from src.twin.config import build, load_hall_config, load_twin_config


def test_shipped_configs_load(paths):
    twin, geom, cfg = build(paths["hall_a"], paths["twin"])
    assert geom.name == "hall_a"
    assert geom.n_racks == 200
    assert geom.n_cracs == 4
    assert cfg.limits.inlet_recommended_max_c == 27.0


def test_unknown_key_in_twin_config_is_rejected(tmp_path):
    p = tmp_path / "twin.yaml"
    p.write_text("recirculation:\n  escap_base: 0.3\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_twin_config(p)


def test_unknown_top_level_section_is_rejected(tmp_path):
    p = tmp_path / "twin.yaml"
    p.write_text("recirulation:\n  escape_base: 0.3\n")
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_twin_config(p)


def test_unknown_key_in_hall_config_is_rejected(tmp_path, hall_cfg):
    import yaml
    p = tmp_path / "hall.yaml"
    p.write_text(yaml.safe_dump(dict(hall_cfg, n_rows=4)))
    with pytest.raises(ValueError, match="unknown key"):
        load_hall_config(p)


def test_missing_required_hall_key_is_rejected(tmp_path, hall_cfg):
    import yaml
    bad = {k: v for k, v in hall_cfg.items() if k != "aisle_width_m"}
    p = tmp_path / "hall.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="missing required key"):
        load_hall_config(p)


def test_hall_may_override_recirculation(tmp_path, paths, hall_cfg):
    import yaml
    p = tmp_path / "hall.yaml"
    p.write_text(yaml.safe_dump(dict(hall_cfg, recirculation={"escape_base": 0.45})))
    twin, _, cfg = build(p, paths["twin"])
    assert cfg.recirculation.escape_base == 0.45
    # Everything not overridden still comes from the twin config.
    assert cfg.recirculation.decay_length_m == 2.5


def test_bad_hall_override_key_is_rejected(tmp_path, paths, hall_cfg):
    import yaml
    p = tmp_path / "hall.yaml"
    p.write_text(yaml.safe_dump(dict(hall_cfg, recirculation={"escape_bass": 0.45})))
    with pytest.raises(ValueError, match="unknown key"):
        build(p, paths["twin"])


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_twin_config(tmp_path / "nope.yaml")
