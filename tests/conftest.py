import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HALL_A = ROOT / "configs" / "hall" / "hall_a.yaml"
TWIN_DEFAULT = ROOT / "configs" / "twin" / "default.yaml"


@pytest.fixture
def hall_cfg():
    """A small paired-layout hall. Small so tests stay fast, but structurally
    identical to hall_a: even rows face +y, CRACs at hot-aisle ends."""
    return {
        "name": "test_hall",
        "rows": 6,
        "racks_per_row": 8,
        "orientation_pattern": "paired",
        "rack_width_m": 0.6,
        "rack_depth_m": 1.2,
        "aisle_width_m": 1.2,
        "crac_units": [{"aisle": 1, "end": "left"}, {"aisle": 3, "end": "right"}],
    }


@pytest.fixture
def paths():
    return {"hall_a": HALL_A, "twin": TWIN_DEFAULT}
