"""Одноразовая генерация новых тестовых CSV (запускать из корня проекта).

12/13/14 строятся на реальных значениях 01_valid_336h.csv: в каждом файле
нарушено ровно одно условие. same_name — пара файлов с одинаковым именем и
размером, но разным содержимым.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_csv"

src = (SAMPLES / "01_valid_336h.csv").read_bytes()
lines = src.decode("utf-8").strip().splitlines()
header, rows = lines[0], lines[1:]
assert rows[0] == "2025-12-18 00:00:00,4637", rows[0]

# 12: дробное значение в первой строке.
rows12 = ["2025-12-18 00:00:00,4637.5"] + rows[1:]
(SAMPLES / "12_fractional_demand.csv").write_text(
    header + "\n" + "\n".join(rows12) + "\n", encoding="utf-8"
)

# 13: бесконечное значение в первой строке.
rows13 = ["2025-12-18 00:00:00,inf"] + rows[1:]
(SAMPLES / "13_infinite_demand.csv").write_text(
    header + "\n" + "\n".join(rows13) + "\n", encoding="utf-8"
)

# 14: все demand записаны строками с пробелами.
rows14 = []
for r in rows:
    ts, d = r.split(",")
    rows14.append(f'{ts}," {d} "')
(SAMPLES / "14_numeric_strings.csv").write_text(
    header + "\n" + "\n".join(rows14) + "\n", encoding="utf-8"
)

# Пара: одинаковое имя и размер, разное содержимое.
a = src
b = src.replace(b",4637", b",9637", 1)
assert len(a) == len(b), (len(a), len(b))
assert a != b
same = ROOT / "same_name"
(same / "variant_a").mkdir(parents=True, exist_ok=True)
(same / "variant_b").mkdir(parents=True, exist_ok=True)
(same / "variant_a" / "sample.csv").write_bytes(a)
(same / "variant_b" / "sample.csv").write_bytes(b)

print("OK", len(a), "bytes")
print("a:", hashlib.sha256(a).hexdigest()[:16])
print("b:", hashlib.sha256(b).hexdigest()[:16])
