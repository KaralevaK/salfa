"""Streamlit-приложение: почасовой прогноз спроса на NYC Yellow Taxi.

Интерфейс построен по мотивам макета forecast_playground_nyc_taxi из архива
Google Stitch и дизайн-системы Urban Kinetic Forecast Studio (DESIGN.md).

Используются только реальные данные проекта: обученная модель LightGBM
(best_model.pkl) и почасовая история (history.csv). Никаких вымышленных
показателей, районов, погоды или доверительных интервалов здесь нет —
модель выдаёт точечный прогноз.

Источник истории можно выбрать: встроенный демо-файл history.csv либо
пользовательский CSV с колонками pickup_hour и demand.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    HISTORY_PATH,
    MODEL_PATH,
    load_history,
    load_model,
    recursive_forecast,
    validate_history,
)

# --- Палитра дизайн-системы -------------------------------------------------
BG = "#090D16"
CARD = "#0F172A"
CARD_BORDER = "#1E293B"
YELLOW = "#FFC72C"
CYAN = "#38BDF8"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"

HORIZON_LABELS = {
    "24 часа": 24,
    "48 часов": 48,
    "7 дней (168 часов)": 168,
}

SOURCE_LABELS = {
    "Демо-данные (history.csv)": "demo",
    "Загрузить CSV": "upload",
}

HISTORY_DAYS_SHOWN = 7  # окно на графике — последние 7 дней факта

st.set_page_config(
    page_title="NYC Yellow Taxi Demand Forecast",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
        html, body, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background-color: {BG};
        }}
        [data-testid="stSidebar"] {{ background-color: {CARD}; }}

        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; }}

        h1, h2, h3 {{ color: {TEXT} !important; }}
        p, li, label {{ color: {TEXT}; }}

        [data-testid="stMetric"] {{
            background-color: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {MUTED};
            font-family: "JetBrains Mono", monospace;
            font-size: 11px;
            line-height: 1.4;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            white-space: normal;
            word-break: break-word;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT};
            font-family: "JetBrains Mono", monospace;
            font-size: 18px;
            line-height: 1.35;
            font-weight: 700;
            white-space: normal;
            word-break: break-word;
        }}
        [data-testid="stMetricDelta"] {{ color: {YELLOW}; }}
        /* Карточки метрик одного размера: растягиваем карточку на всю
           высоту своей колонки, чтобы подписи в разное число строк
           не делали карточки разной высоты. */
        [data-testid="stColumn"]:has([data-testid="stMetric"]) {{
            display: flex;
            flex-direction: column;
        }}
        [data-testid="stColumn"]:has([data-testid="stMetric"])
        > [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto;
        }}
        [data-testid="stColumn"]:has([data-testid="stMetric"])
        [data-testid="stElementContainer"] {{
            flex: 1 1 auto;
            display: flex;
        }}
        [data-testid="stColumn"]:has([data-testid="stMetric"])
        [data-testid="stMetric"] {{
            flex: 1 1 auto;
            display: flex;
        }}
        /* Подпись прижата к верху, число — к низу карточки: содержимое
           всех карточек выровнено одинаково, лишнее место — посередине. */
        [data-testid="stColumn"]:has([data-testid="stMetric"])
        [data-testid="stMetric"] > div {{
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        /* Streamlit задаёт nowrap и размер шрифта внутри метрик — переносим
           текст и уменьшаем подпись, чтобы всё помещалось в карточку. */
        [data-testid="stMetricLabel"] p {{
            font-size: 11px !important;
            line-height: 1.4;
            letter-spacing: 0.03em;
        }}
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"] p {{
            white-space: normal !important;
            word-break: break-word !important;
        }}

        [data-testid="stButton"] > button[kind="primary"] {{
            background-color: {YELLOW};
            color: {BG};
            border: none;
            border-radius: 4px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
        }}
        [data-testid="stButton"] > button[kind="primary"]:hover {{
            background-color: #F5A623;
            color: {BG};
            border: 1px solid #F5A623;
        }}
        [data-testid="stDownloadButton"] > button {{
            background-color: transparent;
            color: {TEXT};
            border: 1px solid #334155;
            border-radius: 4px;
        }}
        [data-testid="stDownloadButton"] > button:hover {{
            background-color: {CARD_BORDER};
            border-color: {MUTED};
            color: {TEXT};
        }}

        [data-testid="stRadio"] > div {{
            background-color: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 4px;
            padding: 4px;
            gap: 4px;
        }}
        [data-testid="stRadio"] label {{ color: {TEXT}; }}

        [data-testid="stAlert"] {{
            background-color: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
        }}

        .ty-card {{
            background-color: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
            padding: 14px 18px;
            height: 100%;
        }}
        .ty-card-label {{
            color: {MUTED};
            font-family: "JetBrains Mono", monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
            /* Место под две строки: подписи в одну и две строки
               не сдвигают значения относительно соседних карточек. */
            min-height: 36px;
        }}
        .ty-card-value {{
            color: {TEXT};
            font-family: "JetBrains Mono", monospace;
            font-size: 22px;
            font-weight: 700;
        }}
        .ty-card-note {{
            color: {YELLOW};
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
            margin-top: 4px;
        }}
        /* Карточки контекста одного размера: height:100% у .ty-card
           срабатывает только при flex-цепочке до колонки — выравниваем
           все карточки по самой высокой. */
        [data-testid="stColumn"]:has(.ty-card) {{
            display: flex;
            flex-direction: column;
        }}
        [data-testid="stColumn"]:has(.ty-card)
        > [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto;
        }}
        [data-testid="stColumn"]:has(.ty-card) [data-testid="stElementContainer"],
        [data-testid="stColumn"]:has(.ty-card) [data-testid="stMarkdown"],
        [data-testid="stColumn"]:has(.ty-card) [data-testid="stMarkdown"] > div,
        [data-testid="stColumn"]:has(.ty-card)
        [data-testid="stMarkdown"] > div > [data-testid="stMarkdownContainer"] {{
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }}
        .ty-card {{ flex: 1 1 auto; }}
        div[data-testid="stExpander"] {{
            background-color: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
        }}

        /* --- Таблицы (markdown) ------------------------------------------- */
        [data-testid="stMarkdownContainer"] table {{
            background-color: {BG};
            color: {TEXT};
            border-collapse: collapse;
            width: 100%;
            font-family: "JetBrains Mono", monospace;
            font-size: 13px;
            margin: 10px 0;
        }}
        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td {{
            border: 1px solid {CARD_BORDER};
            padding: 8px 12px;
            color: {TEXT};
            text-align: left;
            line-height: 1.45;
        }}
        [data-testid="stMarkdownContainer"] th {{
            background-color: {CARD};
            color: {MUTED};
            font-weight: 600;
        }}
        [data-testid="stMarkdownContainer"] tbody tr:nth-child(even) td {{
            background-color: rgba(148, 163, 184, 0.06);
        }}
        [data-testid="stMarkdownContainer"] tbody tr:hover td {{
            background-color: rgba(255, 199, 44, 0.08);
        }}

        /* --- Таблица почасового прогноза (st.dataframe) -------------------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
            overflow: hidden;
        }}

        /* Загрузчик файла */
        [data-testid="stFileUploader"] {{
            background-color: {CARD};
            border: 1px solid {CARD_BORDER};
            border-radius: 8px;
            padding: 6px;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: rgba(255, 255, 255, 0.02);
            border: 1.5px dashed #334155;
            border-radius: 8px;
            transition: border-color 0.2s ease, background-color 0.2s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {YELLOW};
            background-color: rgba(255, 199, 44, 0.04);
        }}
        [data-testid="stFileUploaderDropzoneInstructions"] {{
            color: {MUTED};
            font-family: "JetBrains Mono", monospace;
            font-size: 12px;
        }}
        [data-testid="stFileUploader"] button[kind="secondary"] {{
            background-color: {YELLOW};
            color: {BG};
            border: none;
            border-radius: 4px;
            font-weight: 600;
        }}
        [data-testid="stFileUploader"] button[kind="secondary"]:hover {{
            background-color: #F5A623;
            color: {BG};
            border: 1px solid #F5A623;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Загрузка ресурсов ------------------------------------------------------
@st.cache_resource
def get_model():
    """Модель LightGBM загружается один раз на процесс."""
    return load_model(MODEL_PATH)


@st.cache_data
def get_history() -> pd.DataFrame:
    """История кэшируется: чтение и валидация выполняются при первом запуске."""
    return load_history(HISTORY_PATH)


def fmt_int(value: float) -> str:
    """Форматирует число с пробелом в качестве разделителя тысяч."""
    return f"{value:,.0f}".replace(",", " ")


def forecast_to_csv(forecast: pd.DataFrame) -> str:
    """Готовит CSV с прогнозом для кнопки скачивания."""
    out = forecast.copy()
    out["pickup_hour"] = out["pickup_hour"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["forecast"] = out["forecast"].round(2)
    return out[["pickup_hour", "forecast"]].to_csv(index=False)


def make_sample_csv() -> str:
    """Генерирует пример корректного CSV для загрузки.

    Это синтетический почасовой ряд длиной 200 часов (больше минимальных
    168) с суточной сезонностью — только чтобы показать ожидаемую структуру
    файла. Для реального прогноза используйте фактические данные.
    """
    idx = pd.date_range("2025-11-01 00:00", periods=200, freq="h")
    hours = idx.hour.to_numpy()
    base = 3000 + 2500 * np.sin(2 * np.pi * (hours - 6) / 24)
    noise = np.random.default_rng(42).normal(0, 300, len(idx))
    demand = np.maximum(base + noise, 0).round(0).astype(int)
    df = pd.DataFrame(
        {
            "pickup_hour": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "demand": demand,
        }
    )
    return df.to_csv(index=False)


def make_info_card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="ty-card-note">{note}</div>' if note else ""
    return (
        f'<div class="ty-card"><div class="ty-card-label">{label}</div>'
        f'<div class="ty-card-value">{value}</div>{note_html}</div>'
    )


def build_figure(history_tail: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """График: последние 7 дней факта + прогноз с чёткой границей."""
    fig = go.Figure()

    # Фактическая история
    fig.add_trace(
        go.Scatter(
            x=history_tail["pickup_hour"],
            y=history_tail["demand"],
            name="Факт",
            mode="lines",
            line=dict(color=CYAN, width=2),
            hovertemplate="%{x|%d.%m %H:00}<br>Факт: %{y:,.0f}<extra></extra>",
        )
    )

    # Прогноз. Первым значением берём последний известный факт,
    # чтобы линия прогноза визуально starts от границы.
    boundary_ts = history_tail["pickup_hour"].iloc[-1]
    boundary_val = history_tail["demand"].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=pd.concat(
                [pd.Series([boundary_ts]), forecast["pickup_hour"]],
                ignore_index=True,
            ),
            y=pd.concat(
                [pd.Series([boundary_val]), forecast["forecast"]],
                ignore_index=True,
            ),
            name="Прогноз LightGBM",
            mode="lines",
            line=dict(color=YELLOW, width=2.5, dash="dash"),
            hovertemplate="%{x|%d.%m %H:00}<br>Прогноз: %{y:,.0f}<extra></extra>",
        )
    )

    # Граница между историей и прогнозом
    fig.add_vline(
        x=boundary_ts,
        line_width=1.2,
        line_dash="dot",
        line_color=MUTED,
    )
    fig.add_annotation(
        x=boundary_ts,
        y=1.03,
        yref="paper",
        text="Граница прогноза",
        showarrow=False,
        font=dict(color=MUTED, size=11, family="JetBrains Mono, monospace"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, sans-serif", size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        height=460,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=12),
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            linecolor=CARD_BORDER,
            tickfont=dict(color=MUTED, family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            title="Поездок в час",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            linecolor=CARD_BORDER,
            tickfont=dict(color=MUTED, family="JetBrains Mono, monospace"),
        ),
    )
    return fig


# --- Шапка ------------------------------------------------------------------
st.markdown(
    f"""
    <h1 style="margin-bottom:0;">🚕 NYC Yellow Taxi Demand Forecast</h1>
    <p style="color:{MUTED}; font-size:15px; margin-top:8px; max-width:900px;">
        Почасовой прогноз спроса на NYC Yellow Taxi на следующие 24 часа,
        48 часов или 7 дней на основе модели LightGBM.
    </p>
    """,
    unsafe_allow_html=True,
)

# --- Загрузка ресурсов ------------------------------------------------------
try:
    model = get_model()
except FileNotFoundError as exc:
    st.error(f"Не найден файл модели: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Ошибка загрузки модели: {exc}")
    st.stop()


# --- Источник данных --------------------------------------------------------
source_label = st.radio(
    "Источник данных",
    options=list(SOURCE_LABELS),
    index=0,
    horizontal=True,
)
source = SOURCE_LABELS[source_label]

uploaded = None
history = None
data_error = None

if source == "demo":
    try:
        history = get_history()
    except Exception as exc:  # noqa: BLE001
        data_error = str(exc)
else:
    uploaded = st.file_uploader(
        "Загрузите CSV с колонками pickup_hour и demand",
        type=["csv"],
        help=(
            "Почасовой агрегированный спрос: непрерывный ряд минимум 168 часов, "
            "без пропусков и дубликатов, неотрицательные значения."
        ),
    )
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            warnings = validate_history(raw)
            history = raw.copy()
            for wmsg in warnings:
                st.warning(wmsg)
        except Exception as exc:  # noqa: BLE001
            data_error = str(exc)

    st.download_button(
        label="⬇ Скачать пример CSV",
        data=make_sample_csv(),
        file_name="sample_history.csv",
        mime="text/csv",
    )

# Смена источника данных или загрузка нового файла сбрасывает старый прогноз.
# Ключ строится по SHA-256 содержимого: два файла с одинаковыми именем и
# размером, но разным содержимым не оставят старый прогноз на экране.
if source == "demo":
    data_key = "demo"
elif uploaded is not None:
    data_key = f"upload:{hashlib.sha256(uploaded.getvalue()).hexdigest()}"
else:
    data_key = "upload:none"

if st.session_state.get("data_key") != data_key:
    st.session_state.pop("forecast", None)
    st.session_state["data_key"] = data_key

# --- Карточки контекста -----------------------------------------------------
if data_error is not None:
    st.error(f"Ошибка в данных: {data_error}")

if history is None:
    if source == "upload":
        st.info(
            "Загрузите CSV с колонками *pickup_hour* и *demand* — либо скачайте "
            "пример, чтобы увидеть ожидаемую структуру файла."
        )
    st.stop()

last_hour = pd.Timestamp(history["pickup_hour"].iloc[-1])
first_hour = pd.Timestamp(history["pickup_hour"].iloc[0])
forecast_start = last_hour + pd.Timedelta(hours=1)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        make_info_card("Первый исторический час", f"{first_hour:%d.%m.%Y %H}:00"),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        make_info_card("Последний исторический час", f"{last_hour:%d.%m.%Y %H}:00"),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        make_info_card(
            "Объём истории",
            f"{len(history)} часов",
            "непрерывный почасовой ряд",
        ),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        make_info_card(
            "Начало прогноза",
            f"{forecast_start:%d.%m.%Y %H}:00",
            "следующий час после истории",
        ),
        unsafe_allow_html=True,
    )

st.write("")

# --- Управление -------------------------------------------------------------
control_left, control_right = st.columns([3, 1])

with control_left:
    horizon_label = st.radio(
        "Горизонт прогноза",
        options=list(HORIZON_LABELS),
        index=0,
        horizontal=True,
    )
horizon = HORIZON_LABELS[horizon_label]

with control_right:
    st.write("")  # вертикальное выравнивание кнопки с радио
    st.write("")
    run_clicked = st.button(
        "⚡ Сформировать прогноз",
        type="primary",
        width="stretch",
    )

# Смена горизонта делает ранее полученный результат устаревшим.
if st.session_state.get("horizon") != horizon:
    st.session_state.pop("forecast", None)
    st.session_state["horizon"] = horizon

if run_clicked:
    try:
        with st.spinner("Считаю рекурсивный прогноз..."):
            result = recursive_forecast(model, history, horizon)
        st.session_state["forecast"] = result
        st.session_state["horizon"] = horizon
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось построить прогноз: {exc}")

# --- Результаты -------------------------------------------------------------
forecast = st.session_state.get("forecast")

if forecast is None:
    st.info(
        "Выберите горизонт и нажмите «Сформировать прогноз». "
        f"Данные истории заканчиваются {last_hour:%d.%m.%Y %H}:00 — "
        "прогноз начнётся со следующего часа."
    )
else:
    if horizon == 168:
        st.warning(
            "Это рекурсивный прогноз на 168 часов (7 дней): каждое значение "
            "подставляется в расчёт следующего часа, поэтому ошибки "
            "накапливаются, и неопределённость растёт с увеличением горизонта. "
            "Отдалённые часы стоит воспринимать как ориентир, а не как "
            "точное значение."
        )

    # --- Карточки метрик ---
    total = forecast["forecast"].sum()
    average = forecast["forecast"].mean()
    peak_value = forecast["forecast"].max()
    peak_time = forecast.loc[forecast["forecast"].idxmax(), "pickup_hour"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Средний прогнозируемый спрос", f"{fmt_int(average)} поездок/ч")
    with m2:
        st.metric("Максимальный спрос", f"{fmt_int(peak_value)} поездок/ч")
    with m3:
        st.metric("Пик прогноза", f"{peak_time:%d.%m %H}:00")
    with m4:
        st.metric(f"Суммарно за {horizon_label}", f"{fmt_int(total)} поездок")

    st.write("")

    # --- График ---
    history_tail = history.tail(HISTORY_DAYS_SHOWN * 24)
    st.plotly_chart(
        build_figure(history_tail, forecast),
        width="stretch",
        config={"displayModeBar": False},
    )

    st.write("")

    # --- Таблица и скачивание ---
    table_left, table_right = st.columns([3, 1])

    with table_left:
        st.subheader("Почасовой прогноз")
        table = forecast.copy()
        table["Час"] = table["pickup_hour"].dt.strftime("%d.%m.%Y %H:00")
        table["Прогноз спроса, поездок/ч"] = table["forecast"].round(0).astype(int)
        st.dataframe(
            table[["Час", "Прогноз спроса, поездок/ч"]],
            hide_index=True,
            width="stretch",
        )

    with table_right:
        st.write("")
        st.write("")
        csv_bytes = forecast_to_csv(forecast)
        st.download_button(
            label="⬇ Скачать CSV",
            data=csv_bytes,
            file_name=f"yellow_taxi_forecast_{horizon}h.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("О модели и качестве"):
        st.markdown(
            f"""
            **Модель.** Используется градиентный бустинг **LightGBM**
            (`LGBMRegressor`). Модель опирается на календарные признаки
            (год, месяц, день недели, час, день года, выходной), лаги
            `lag_1, lag_2, lag_3, lag_24, lag_48, lag_168` и скользящие
            статистики (mean/median за 24 и 168 часов) — всего 22 признака.

            **Как строится прогноз.** Модель предсказывает один час, поэтому
            горизонт 24, 48 или 168 часов получается рекурсивно: каждое
            предсказанное значение подставляется в расчёт следующего часа.
            Окна статистик заканчиваются на часе `t-1`, то есть используют
            только уже известные данные. Отрицательные прогнозы обрезаются
            до нуля.

            **Пользовательские данные.** Вместо демо-истории можно загрузить
            свой CSV с колонками `pickup_hour` (дата и час) и `demand`
            (почасовой спрос). Это должен быть почасовой агрегированный спрос
            NYC Yellow Taxi со структурой, сопоставимой с данными, на которых
            обучалась модель: непрерывный ряд минимум 168 часов, без пропусков
            и дубликатов, неотрицательные целые значения. Приложение само
            рассчитает все 22 признака из загруженной истории.

            **Качество на независимой Test-выборке** (из исходного notebook
            обучения модели):

            | Сценарий | MAPE | MAE, поездок/час | RMSE, поездок/час |
            |---|---|---|---|
            | Прогноз следующего часа (one-step) | 6.81% | ≈ 272 | 406.88 |
            | Рекурсивный прогноз на 24 часа | 12.36% | ≈ 514 | 784.67 |

            Сценарий «рекурсивный прогноз на 24 часа» наиболее близок к тому,
            как работает это приложение при выборе горизонта 24 часа.

            **Важно.** При увеличении горизонта ошибки могут накапливаться,
            поэтому для 48 и 168 часов отдельная оценка на Test не
            проводилась, и показатели 24 часа на них не переносятся.
            Приведённые метрики — результат на тестовой выборке из исходного
            исследования, а **не гарантия** точности будущих прогнозов.
            """
        )
