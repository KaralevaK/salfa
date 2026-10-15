"""Автоматические тесты ядра прогнозирования (pytest).

Запуск из корня проекта:

    python -m pytest tests/test_forecasting.py -v

Что проверяется на реальных best_model.pkl и history.csv:

- загрузка модели и истории;
- прогнозы на горизонтах 24, 48 и 168 часов;
- точное количество прогнозируемых часов;
- отсутствие NaN и бесконечных значений;
- неотрицательность спроса;
- непрерывность почасовых временных отметок;
- начало прогноза сразу после последнего часа истории;
- неизменность входной истории после прогнозирования;
- воспроизводимость результата при повторном запуске.

Тесты детерминированы: не используют текущую дату и не требуют интернета.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting import (  # noqa: E402
    FORECAST_HORIZONS,
    load_history,
    load_model,
    recursive_forecast,
)

HORIZONS = list(FORECAST_HORIZONS)


@pytest.fixture(scope="session")
def model():
    """Обученная модель best_model.pkl."""
    return load_model()


@pytest.fixture(scope="session")
def history() -> pd.DataFrame:
    """Валидированная история history.csv."""
    return load_history()


@pytest.fixture(scope="session")
def forecasts(history, model) -> dict[int, pd.DataFrame]:
    """Прогнозы на все поддерживаемые горизонты (считаются один раз)."""
    return {h: recursive_forecast(model, history, h) for h in HORIZONS}


# == Загрузка модели и истории ==


def test_model_loads(model) -> None:
    assert hasattr(model, "predict"), "у модели нет метода predict"


def test_history_loads(history) -> None:
    assert list(history.columns) == ["pickup_hour", "demand"]
    assert len(history) >= 168, "истории меньше минимума в 168 часов"
    assert pd.api.types.is_datetime64_any_dtype(history["pickup_hour"])
    assert pd.api.types.is_numeric_dtype(history["demand"])


def test_supported_horizons() -> None:
    assert HORIZONS == [24, 48, 168]


# == Свойства прогнозов на каждом горизонте ==


@pytest.mark.parametrize("horizon", HORIZONS)
def test_exact_number_of_hours(forecasts, horizon) -> None:
    fc = forecasts[horizon]
    assert len(fc) == horizon
    assert list(fc.columns) == ["pickup_hour", "forecast", "is_forecast"]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_no_nan_or_infinite(forecasts, horizon) -> None:
    values = forecasts[horizon]["forecast"].to_numpy(dtype=float)
    assert not np.isnan(values).any(), "в прогнозе есть NaN"
    assert np.isfinite(values).all(), "в прогнозе есть бесконечные значения"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_demand_non_negative(forecasts, horizon) -> None:
    values = forecasts[horizon]["forecast"].to_numpy(dtype=float)
    assert (values >= 0).all(), "прогноз содержит отрицательный спрос"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_hourly_timestamps_continuous(history, forecasts, horizon) -> None:
    fc = forecasts[horizon]
    steps = fc["pickup_hour"].diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all(), "нарушена почасовая непрерывность"
    # Первый прогнозный час идёт сразу за последним часом истории.
    last_history = pd.Timestamp(history["pickup_hour"].iloc[-1])
    assert fc["pickup_hour"].iloc[0] == last_history + pd.Timedelta(hours=1)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_forecast_starts_after_history(history, forecasts, horizon) -> None:
    fc = forecasts[horizon]
    last_history = pd.Timestamp(history["pickup_hour"].iloc[-1])
    assert fc["pickup_hour"].min() > last_history, (
        "прогноз начинается не после последнего часа истории"
    )
    assert fc["pickup_hour"].max() == last_history + pd.Timedelta(hours=horizon)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_all_rows_are_forecast(forecasts, horizon) -> None:
    assert forecasts[horizon]["is_forecast"].all()


# == Неизменность входа и воспроизводимость ==


def test_history_not_modified_by_forecast(model) -> None:
    df = load_history()
    before = df.copy(deep=True)
    recursive_forecast(model, df, 24)
    pd.testing.assert_frame_equal(df, before)


def test_reproducible_with_same_input(history, model) -> None:
    first = recursive_forecast(model, history.copy(deep=True), 24)
    second = recursive_forecast(model, history.copy(deep=True), 24)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert (
        first["forecast"].to_numpy() == second["forecast"].to_numpy()
    ).all(), "повторный запуск дал другие значения"
