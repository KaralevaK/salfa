from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Настройки страницы
# ---------------------------------------------------------

st.set_page_config(
    page_title="NYC Yellow Taxi Demand Forecast",
    page_icon="🚕",
    layout="wide"
)


# ---------------------------------------------------------
# Загрузка модели и истории
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner=False)
def load_model():
    model_file = open(BASE_DIR / 'best_model.pkl', 'rb')
    model = joblib.load(model_file)
    return model


@st.cache_data
def load_history():
    df = pd.read_csv(
        BASE_DIR / "history.csv",
        index_col=0,
        parse_dates=True
    )

    df.index = pd.to_datetime(df.index)

    return df.sort_index()


model = load_model()
history = load_history()


# ---------------------------------------------------------
# Признаки модели
# ---------------------------------------------------------

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
    "rolling_median_168"
]


# ---------------------------------------------------------
# Формирование признаков для одного будущего часа
# ---------------------------------------------------------

def make_features(timestamp, demand_history):

    row = {}

    # Календарные признаки
    row["year"] = timestamp.year
    row["month"] = timestamp.month
    row["day_of_week"] = timestamp.dayofweek
    row["hour"] = timestamp.hour
    row["day_of_year"] = timestamp.dayofyear

    row["is_weekend"] = int(
        timestamp.dayofweek >= 5
    )

    # Циклические признаки
    row["hour_sin"] = np.sin(
        2 * np.pi * timestamp.hour / 24
    )

    row["hour_cos"] = np.cos(
        2 * np.pi * timestamp.hour / 24
    )

    row["dow_sin"] = np.sin(
        2 * np.pi * timestamp.dayofweek / 7
    )

    row["dow_cos"] = np.cos(
        2 * np.pi * timestamp.dayofweek / 7
    )

    row["year_sin"] = np.sin(
        2 * np.pi * timestamp.dayofyear / 365.25
    )

    row["year_cos"] = np.cos(
        2 * np.pi * timestamp.dayofyear / 365.25
    )

    # Лаговые признаки
    row["lag_1"] = demand_history.iloc[-1]
    row["lag_2"] = demand_history.iloc[-2]
    row["lag_3"] = demand_history.iloc[-3]

    row["lag_24"] = demand_history.iloc[-24]
    row["lag_48"] = demand_history.iloc[-48]
    row["lag_168"] = demand_history.iloc[-168]

    # Rolling-признаки
    row["rolling_mean_24"] = (
        demand_history.iloc[-24:].mean()
    )

    row["rolling_median_24"] = (
        demand_history.iloc[-24:].median()
    )

    row["rolling_mean_168"] = (
        demand_history.iloc[-168:].mean()
    )

    row["rolling_median_168"] = (
        demand_history.iloc[-168:].median()
    )

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS
    )


# ---------------------------------------------------------
# Рекурсивный прогноз
# ---------------------------------------------------------

def recursive_forecast(model, history, horizon):

    demand_history = (
        history["demand"]
        .astype(float)
        .copy()
    )

    last_timestamp = demand_history.index[-1]

    predictions = []

    for step in range(1, horizon + 1):

        forecast_time = (
            last_timestamp
            + pd.Timedelta(hours=step)
        )

        X_future = make_features(
            forecast_time,
            demand_history
        )

        prediction = model.predict(X_future)[0]

        predictions.append({
            "datetime": forecast_time,
            "predicted_demand": prediction
        })

        # Добавляем прогноз в историю.
        # Следующий прогноз сможет использовать его как lag.
        demand_history.loc[forecast_time] = prediction

    return pd.DataFrame(predictions)


# ---------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------

st.title("🚕 NYC Yellow Taxi Demand Forecast")

st.write(
    "Прогноз почасового спроса на Yellow Taxi "
    "с использованием модели LightGBM."
)

last_date = history.index.max()

st.info(
    f"Последнее доступное наблюдение: "
    f"{last_date:%d.%m.%Y %H:%M}"
)


horizon = st.slider(
    "Горизонт прогноза, часов",
    min_value=1,
    max_value=24,
    value=24
)


if st.button("Построить прогноз"):

    forecast = recursive_forecast(
        model=model,
        history=history,
        horizon=horizon
    )

    st.subheader(
        f"Прогноз на следующие {horizon} ч."
    )

    # Таблица для отображения
    forecast_display = forecast.copy()

    forecast_display["predicted_demand"] = (
        forecast_display["predicted_demand"]
        .round()
        .astype(int)
    )

    forecast_display = forecast_display.rename(
        columns={
            "datetime": "Дата и время",
            "predicted_demand": "Прогноз поездок"
        }
    )

    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True
    )

    # График
    chart_data = (
        forecast
        .set_index("datetime")[
            ["predicted_demand"]
        ]
    )

    st.line_chart(chart_data)