"""Ядро прогнозирования спроса на NYC Yellow Taxi.

Модель best_model.pkl — LightGBMRegressor, который ожидает ровно 22 признака
в строго определённом порядке. Все признаки — это календарные (циклические и
числовые) и авторегрессионные (lag / rolling) переменные, поэтому прогноз
на любой горизонт строится рекурсивно: по одному часу вперёд, с подстановкой
каждого полученного значения обратно во временную серию.

Прогнозирование ведётся по позиционной индексации непрерывного почасового
ряда: окна rolling заканчиваются на часе t-1 и никогда не захватывают
прогнозируемый момент, то есть data leakage исключён.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
HISTORY_PATH = BASE_DIR / "history.csv"

REQUIRED_COLUMNS = ["pickup_hour", "demand"]
MIN_HISTORY_HOURS = 168
FORECAST_HORIZONS = (24, 48, 168)

# Порядок признаков, который модель получала при обучении. Менять нельзя.
FEATURE_COLUMNS = [
    "year",
    "month",
    "day_of_week",
    "hour",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_median_24",
    "rolling_mean_168",
    "rolling_median_168",
]

LAG_HOURS = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS = (24, 168)

LAG_OFFSETS = {1: 1, 2: 2, 3: 3, 24: 24, 48: 48, 168: 168}


def load_model(path: Path = MODEL_PATH):
    """Загружает обученную модель LightGBM."""
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    return joblib.load(path)


def load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Читает историю спроса, валидирует и очищает её.

    Возвращает отсортированный по времени DataFrame с колонками
    pickup_hour (datetime) и demand (числовой).
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл истории не найден: {path}")

    df = pd.read_csv(path)
    validate_history(df)
    return df


def validate_history(df: pd.DataFrame) -> list[str]:
    """Валидирует и очищает входные данные (почасовую историю спроса).

    Автоматические действия (данные по существу не изменяются):
    - сортировка строк по pickup_hour;
    - удаление полностью одинаковых строк (pickup_hour и demand совпадают),
      оставляется одна запись на час;
    - приведение demand к числовому типу после проверок (строковые значения,
      корректно преобразующиеся в числа, допустимы).

    Возвращает список предупреждений (например, о числе удалённых дубликатов).

    Поднимает ValueError с понятным сообщением, если данные непригодны:
    - demand не является конечными неотрицательными целыми числами;
    - одинаковый pickup_hour с разным demand (конфликт) — не удаляем и не
      агрегируем автоматически, указываем проблемный час;
    - пропущенные часы (ряд не непрерывен) — не заполняем автоматически;
    - менее 168 последовательных часов после очистки.
    """
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Во входных данных отсутствуют обязательные колонки: "
            + ", ".join(missing)
            + f". Найдены: {', '.join(map(str, df.columns))}"
        )

    if len(df) == 0:
        raise ValueError("Входные данные пусты.")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        bad = df[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(
            "Во входных данных обнаружены пропуски: "
            + ", ".join(f"{col}={count}" for col, count in bad.items())
        )

    # Строковые значения, корректно преобразующиеся в числа, допустимы;
    # после всех проверок demand приводится к числовому типу ниже.
    demand = pd.to_numeric(df["demand"], errors="coerce")
    if demand.isna().any():
        raise ValueError(
            "Колонка demand должна быть числовой: найдены нечисловые значения."
        )
    if not np.isfinite(demand.to_numpy(dtype=float)).all():
        raise ValueError(
            "Колонка demand должна содержать только конечные значения: "
            "найдены бесконечные (inf/-inf)."
        )
    if (demand % 1 != 0).any():
        raise ValueError(
            "Колонка demand должна содержать целые значения: найдены дробные."
        )
    if (demand < 0).any():
        raise ValueError(
            "Колонка demand не должна содержать отрицательных значений "
            "(спрос не может быть отрицательным)."
        )
    # Приведение к числовому типу после проверок.
    df["demand"] = demand

    timestamps = pd.to_datetime(df["pickup_hour"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("В колонке pickup_hour есть нечитаемые даты и время.")

    # Автоматическая сортировка по времени.
    df["pickup_hour"] = timestamps
    df.sort_values("pickup_hour", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Полные дубликаты (pickup_hour И demand совпадают) — удаляем, оставляя
    # одну запись на час.
    full_dup_mask = df.duplicated(subset=["pickup_hour", "demand"], keep="first")
    n_removed = int(full_dup_mask.sum())
    if n_removed > 0:
        df.drop(index=df.index[full_dup_mask], inplace=True)
        df.reset_index(drop=True, inplace=True)
        warnings.append(f"Удалено полностью одинаковых строк: {n_removed}.")

    # Конфликтующие дубликаты (одинаковый pickup_hour, но разный demand) —
    # не удаляем и не агрегируем, показываем ошибку с указанием проблемного
    # часа.
    conflict_mask = df.duplicated(subset=["pickup_hour"], keep=False)
    if conflict_mask.any():
        bad_ts = df.loc[conflict_mask, "pickup_hour"].iloc[0]
        raise ValueError(
            f"Час {bad_ts} встречается несколько раз с разными значениями "
            f"demand. Уточните данные: для каждого часа должно быть одно "
            f"значение спроса."
        )

    # После очистки проверяем минимальную длину истории.
    if len(df) < MIN_HISTORY_HOURS:
        raise ValueError(
            f"Недостаточно истории: нужно минимум {MIN_HISTORY_HOURS} часов "
            f"(для lag_168), найдено {len(df)}."
        )

    # Пропущенные часы не заполняем автоматически — показываем ошибку.
    steps = df["pickup_hour"].diff().dropna()
    bad_steps = steps[steps != pd.Timedelta(hours=1)]
    if len(bad_steps) > 0:
        first = bad_steps.index[0]
        raise ValueError(
            "Временной ряд не непрерывен: найден разрыв в шаге "
            f"{first} (размер {bad_steps.iloc[0]}). "
            "Требуется ровно шаг в 1 час."
        )

    return warnings


def _calendar_features(ts: pd.Timestamp) -> dict:
    """Календарные и циклические (trigonometric encoding) признаки."""
    hour = int(ts.hour)
    dow = int(ts.dayofweek)  # понедельник = 0
    doy = int(ts.dayofyear)

    return {
        "year": int(ts.year),
        "month": int(ts.month),
        "day_of_week": dow,
        "hour": hour,
        "day_of_year": doy,
        "is_weekend": int(dow >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "year_sin": np.sin(2 * np.pi * doy / 365.25),
        "year_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def build_features(ts: pd.Timestamp, values: np.ndarray, pos: int) -> dict:
    """Собирает все 22 признака для одного прогнозируемого часа.

    values — непрерывный почасовой ряд (история + уже построенные прогнозы),
    pos — позиция прогнозируемого часа в этом ряду.

    Окна заканчиваются на pos - 1, то есть на часе, известном до
    прогнозируемого момента.
    """
    if pos < MIN_HISTORY_HOURS:
        raise ValueError(
            f"Недостаточно известных часов перед точкой прогноза: {pos} "
            f"< {MIN_HISTORY_HOURS} (нужно для lag_168)."
        )

    features = _calendar_features(ts)

    for lag, offset in LAG_OFFSETS.items():
        features[f"lag_{lag}"] = float(values[pos - offset])

    for window in ROLLING_WINDOWS:
        # values[pos - window : pos] — это часы t-window ... t-1
        known = values[pos - window : pos]
        features[f"rolling_mean_{window}"] = float(np.mean(known))
        features[f"rolling_median_{window}"] = float(np.median(known))

    return features


def recursive_forecast(
    model,
    history: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Рекурсивно строит прогноз на horizon часов вперёд.

    Каждое предсказанное значение дописывается в ряд перед расчётом
    следующего часа, поэтому лаги и скользящие окна всегда опираются
    на актуальное состояние ряда.
    """
    if horizon not in FORECAST_HORIZONS:
        raise ValueError(
            f"Неподдерживаемый горизонт {horizon}. "
            f"Доступны: {', '.join(map(str, FORECAST_HORIZONS))} часов."
        )

    validate_history(history)

    start = pd.Timestamp(history["pickup_hour"].iloc[0])
    # Непрерывность проверена в validate_history, позиции соответствуют часам.
    values = history["demand"].to_numpy(dtype=float, copy=True)
    last_ts = pd.Timestamp(history["pickup_hour"].iloc[-1])
    history_size = len(values)

    rows = []
    for step in range(1, horizon + 1):
        ts = last_ts + pd.Timedelta(hours=step)
        pos = history_size + step - 1

        # Для pos значения step-1 уже предсказаны и дописаны в values.
        features = build_features(ts, values, pos)
        x = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        prediction = float(model.predict(x)[0])
        # LightGBM может уходить в отрицательные значения на ночных впадинах.
        prediction = max(prediction, 0.0)

        values = np.append(values, prediction)
        rows.append(
            {
                "pickup_hour": ts,
                "forecast": prediction,
                "is_forecast": True,
            }
        )

    return pd.DataFrame(rows)
