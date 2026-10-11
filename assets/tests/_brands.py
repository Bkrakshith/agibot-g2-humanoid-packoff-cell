"""Brand names that must not appear in any asset (labels, prim names, materials).

Extra names can be listed one per line in brand_blocklist.local.txt next to this file. That file is not
committed, so project-specific names stay out of the repository.
"""

from pathlib import Path

_LOCAL = Path(__file__).with_name("brand_blocklist.local.txt")

BRANDS = ["Omron", "Universal Robots", "Fanuc", "KUKA", "Wexxar", "Lantech", "Loveshaw", "Yaskawa", "Doosan",
          "Techman", "Smurfit", "DS Smith"]
if _LOCAL.is_file():
    BRANDS += [line.strip() for line in _LOCAL.read_text().splitlines() if line.strip() and not line.startswith("#")]
