"""Проверка всех тестовых сценариев загрузки пользовательского CSV.

Запуск из корня проекта:

    python tests/run_checks.py

Что проверяется:

- 14 файлов из tests/sample_csv/: положительные проходят валидацию
  (demand приводится к числовому типу) и строят прогноз на 24 часа,
  отрицательные отклоняются с ожидаемым фрагментом сообщения;
- сообщения валидации не содержат упоминаний history.csv;
- два файла с одинаковым именем и размером, но разным содержимым
  (tests/same_name/) дают разные SHA-256 — смена источника в app.py
  определяется по содержимому, а не по имени и размеру;
- app.py действительно использует hashlib.sha256 от содержимого файла.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting import load_model, recursive_forecast, validate_history  # noqa: E402

SAMPLES = Path(__file__).resolve().parent / "sample_csv"
SAME_NAME = Path(__file__).resolve().parent / "same_name"

# (файл, ожидаемый результат, фрагмент ожидаемого сообщения об ошибке)
SCENARIOS: list[tuple[str, str, str | None]] = [
    ("01_valid_336h.csv", "pass", None),
    ("02_valid_168h.csv", "pass", None),
    ("03_invalid_columns.csv", "fail", "обязательные колонки"),
    ("04_insufficient_history.csv", "fail", "Недостаточно истории"),
    ("05_missing_hour.csv", "fail", "не непрерывен"),
    ("06_duplicate_timestamp.csv", "fail", "встречается несколько раз"),
    ("07_missing_demand.csv", "fail", "пропуски"),
    ("08_negative_demand.csv", "fail", "отрицательных"),
    ("09_non_numeric_demand.csv", "fail", "числовой"),
    ("10_full_duplicates.csv", "pass", None),
    ("11_unsorted.csv", "pass", None),
    ("12_fractional_demand.csv", "fail", "целые"),
    ("13_infinite_demand.csv", "fail", "конечные"),
    ("14_numeric_strings.csv", "pass", None),
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail and not ok:
        line += f" — {detail}"
    print(line)
    if not ok:
        failures.append(name)


def run_scenarios() -> dict[str, float]:
    """Возвращает сумму прогноза на 24 часа по положительным файлам."""
    sums: dict[str, float] = {}
    for fname, expected, fragment in SCENARIOS:
        path = SAMPLES / fname
        df = pd.read_csv(path)

        if expected == "pass":
            try:
                warnings = validate_history(df)
            except ValueError as exc:
                check(fname, False, f"неожиданная ошибка: {exc}")
                continue
            numeric = np.issubdtype(df["demand"].dtype, np.number)
            check(f"{fname}: валидация пройдена", True)
            check(f"{fname}: demand числовой тип", numeric, str(df["demand"].dtype))
            if fname == "10_full_duplicates.csv":
                has_warn = any("Удалено полностью одинаковых строк" in w for w in warnings)
                check(f"{fname}: есть предупреждение об удалении", has_warn)
            try:
                fc = recursive_forecast(load_model(), df, 24)
                ok = len(fc) == 24 and np.isfinite(fc["forecast"]).all() \
                    and (fc["forecast"] >= 0).all()
                check(f"{fname}: прогноз 24ч построен (24 строки)", ok)
                sums[fname] = float(fc["forecast"].sum())
            except Exception as exc:  # noqa: BLE001
                check(f"{fname}: прогноз 24ч построен", False, str(exc))
        else:
            try:
                validate_history(df)
            except ValueError as exc:
                msg = str(exc)
                check(
                    f"{fname}: отклонён с ожидаемым сообщением",
                    fragment is not None and fragment in msg,
                    f"получено: {msg}",
                )
                check(
                    f"{fname}: без упоминаний history.csv",
                    "history.csv" not in msg,
                    msg,
                )
            except Exception as exc:  # noqa: BLE001
                check(f"{fname}: отклонён ValueError", False, f"другое исключение: {exc}")
            else:
                check(f"{fname}: отклонён", False, "валидация прошла, ожидалась ошибка")
    return sums


def run_same_name() -> None:
    a_path = SAME_NAME / "variant_a" / "sample.csv"
    b_path = SAME_NAME / "variant_b" / "sample.csv"
    a, b = a_path.read_bytes(), b_path.read_bytes()

    check("same_name: имена файлов совпадают", a_path.name == b_path.name)
    check("same_name: размеры совпадают", len(a) == len(b), f"{len(a)} != {len(b)}")
    check(
        "same_name: SHA-256 различаются",
        hashlib.sha256(a).digest() != hashlib.sha256(b).digest(),
    )


def run_app_wiring() -> None:
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    check(
        "app.py: ключ источника — SHA-256 содержимого",
        "hashlib.sha256(uploaded.getvalue())" in src,
    )
    check("app.py: имя/размер больше не используются", "uploaded.size" not in src)
    check("app.py: импортирован hashlib", "import hashlib" in src)


def main() -> int:
    print("== Сценарии CSV ==")
    sums = run_scenarios()

    print("\n== Одинаковое имя и размер, разное содержимое ==")
    run_same_name()

    print("\n== Связка с app.py ==")
    run_app_wiring()

    if "01_valid_336h.csv" in sums and "14_numeric_strings.csv" in sums:
        same = np.isclose(sums["01_valid_336h.csv"], sums["14_numeric_strings.csv"])
        check(
            "Строковые числа дают тот же прогноз, что и числа",
            same,
            f"{sums['01_valid_336h.csv']} != {sums['14_numeric_strings.csv']}",
        )

    print()
    if failures:
        print(f"ИТОГ: {len(failures)} провалено -> {failures}")
        return 1
    print("ИТОГ: все проверки пройдены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
