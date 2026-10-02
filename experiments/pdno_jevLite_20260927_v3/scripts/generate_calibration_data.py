"""Generate only the calibration role; locked role generation is unavailable here."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.data.generate import generate_roles  # noqa: E402

records = generate_roles(ROOT / "data/manifests/parent_roles_metadata_only.json",
                         ROOT / "data/evaluation_v1", roles=("calibration",))
print(json.dumps(records, indent=2))
