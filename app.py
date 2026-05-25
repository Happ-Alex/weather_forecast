"""
app.py — Streamlit-застосунок для прогнозування опадів
Використовує Open-Meteo API + ML-класифікацію
"""

import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

from data_loader import geocode_city, fetch_historical_data, fetch_forecast_data, save_dataset
from ml_pipeline import (
    prepare_train_data, evaluate_models, get_feature_importance,
    prepare_forecast_features, predict, save_model
)

# ─── Конфігурація сторінки ────────────────────────────────────────────────────
st.set_page_config(
    page_title="️Прогноз опадів",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .rain-yes {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1565c0;
        border: 1px solid #90caf9;
    }
    .rain-no {
        background: linear-gradient(135deg, #fff9c4, #fff176);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #f57f17;
        border: 1px solid #ffe082;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #202124;
        border-left: 4px solid #1a73e8;
        padding-left: 10px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Заголовок ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Прогноз опадів</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Прогнозування опадів на основі даних Open-Meteo та машинного навчання</div>', unsafe_allow_html=True)

# ─── Ініціалізація стану ──────────────────────────────────────────────────────
if "df_history" not in st.session_state:
    st.session_state.df_history = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "best_name" not in st.session_state:
    st.session_state.best_name = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None
if "location_info" not in st.session_state:
    st.session_state.location_info = None
if "lat" not in st.session_state:
    st.session_state.lat = 50.4501
if "lon" not in st.session_state:
    st.session_state.lon = 30.5234
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 1 — ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Блок 1: Завантаження даних з Open-Meteo</div>', unsafe_allow_html=True)

with st.expander("Налаштування завантаження даних", expanded=True):
    col1, col2 = st.columns([2, 1])

    with col1:
        input_mode = st.radio(
            "Спосіб введення місця:",
            ["Назва міста", "Координати вручну"],
            horizontal=True
        )

        if input_mode == "Назва міста":
            city_name = st.text_input(
                "Назва населеного пункту",
                value="Київ",
                placeholder="наприклад: Харків, Одеса, Львів..."
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                manual_lat = st.number_input("Широта (latitude)", value=50.4501, format="%.4f", min_value=-90.0, max_value=90.0)
            with c2:
                manual_lon = st.number_input("Довгота (longitude)", value=30.5234, format="%.4f", min_value=-180.0, max_value=180.0)

    with col2:
        today = date.today()
        default_start = today - timedelta(days=5 * 365)
        start_date = st.date_input("Дата початку", value=default_start, min_value=date(1940, 1, 1), max_value=today - timedelta(days=30))
        end_date = st.date_input("Дата кінця", value=today - timedelta(days=1), min_value=date(1940, 1, 2), max_value=today - timedelta(days=1))

    st.info("💡 Рекомендований період: 3–10 років для кращого навчання моделі.")

    col_btn1, col_btn2, _ = st.columns([1, 1, 2])

    with col_btn1:
        load_btn = st.button("🔄 Завантажити дані з API", type="primary", use_container_width=True)

    with col_btn2:
        uploaded_csv = st.file_uploader("або завантажте CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_csv is not None:
        try:
            df_up = pd.read_csv(uploaded_csv, parse_dates=["date"])
            st.session_state.df_history = df_up
            st.session_state.location_info = "з CSV-файлу"
            st.success(f"CSV завантажено: {len(df_up)} рядків")
        except Exception as e:
            st.error(f"Помилка читання CSV: {e}")

    if load_btn:
        # Визначаємо координати
        if input_mode == "Назва міста":
            with st.spinner(f"Пошук координат для '{city_name}'..."):
                try:
                    lat, lon, display_name = geocode_city(city_name)
                    st.session_state.lat = lat
                    st.session_state.lon = lon
                    st.session_state.location_info = display_name
                    st.info(f"Знайдено: **{display_name}** ({lat:.4f}°, {lon:.4f}°)")
                except Exception as e:
                    st.error(str(e))
                    st.stop()
        else:
            lat, lon = manual_lat, manual_lon
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.location_info = f"({lat:.4f}°, {lon:.4f}°)"

        # Завантажуємо дані
        with st.spinner("Завантаження даних з Open-Meteo..."):
            try:
                df = fetch_historical_data(
                    st.session_state.lat,
                    st.session_state.lon,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )
                st.session_state.df_history = df
                # Скидаємо модель при нових даних
                st.session_state.model_results = None
                st.session_state.best_model = None
                st.session_state.prediction_result = None

                # Зберігаємо CSV
                df.to_csv("weather_daily.csv", index=False)
                st.success(f"Завантажено **{len(df)}** записів для **{st.session_state.location_info}**")
            except Exception as e:
                st.error(f"Помилка завантаження: {e}")
                st.stop()

# Показ датасету
if st.session_state.df_history is not None:
    df = st.session_state.df_history
    rain_pct = (df["precipitation_sum"] > 0).mean() * 100 if "precipitation_sum" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Кількість днів", f"{len(df):,}")
    col2.metric("Від", df["date"].min().strftime("%d.%m.%Y") if "date" in df.columns else "—")
    col3.metric("До", df["date"].max().strftime("%d.%m.%Y") if "date" in df.columns else "—")
    col4.metric("Днів з опадами", f"{rain_pct:.1f}%")

    with st.expander("Переглянути дані"):
        st.dataframe(df.tail(30), use_container_width=True)

    # Завантаження CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Завантажити weather_daily.csv",
        data=csv_bytes,
        file_name="weather_daily.csv",
        mime="text/csv",
    )

    # Графік опадів
    if "precipitation_sum" in df.columns:
        st.markdown("**Кількість опадів за весь період:**")
        fig_rain = px.bar(
            df, x="date", y="precipitation_sum",
            color_discrete_sequence=["#1a73e8"],
            labels={"date": "Дата", "precipitation_sum": "Опади (мм)"},
            height=250,
        )
        fig_rain.update_layout(margin=dict(t=10, b=30), showlegend=False)
        st.plotly_chart(fig_rain, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 2 — НАВЧАННЯ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Блок 2: Навчання та оцінка моделей</div>', unsafe_allow_html=True)

if st.session_state.df_history is None:
    st.warning("Спочатку завантажте дані у Блоці 1.")
else:
    with st.expander("Запуск навчання", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            train_btn = st.button("Навчити моделі", type="primary", use_container_width=True)
        with col2:
            st.caption("Будуть навчені: Random Forest, Gradient Boosting, Logistic Regression, SVM. Відбір ознак відбувається автоматично.")

    if train_btn:
        df = st.session_state.df_history
        with st.spinner("Підготовка ознак і навчання моделей... (може зайняти 1–2 хвилини)"):
            try:
                X, y, feature_cols = prepare_train_data(df)
                results, best_name, best_model, selected_features = evaluate_models(X, y)

                st.session_state.model_results = results
                st.session_state.best_model = best_model
                st.session_state.best_name = best_name
                st.session_state.selected_features = selected_features
                st.session_state.X_train = X
                st.session_state.y_train = y
                st.session_state.feature_cols = feature_cols

                st.success(f"Навчання завершено! Найкраща модель: **{best_name}**")
            except Exception as e:
                st.error(f"Помилка навчання: {e}")
                st.stop()

    if st.session_state.model_results is not None:
        results = st.session_state.model_results
        best_name = st.session_state.best_name

        # Таблиця порівняння моделей
        st.markdown("#### Порівняння моделей (5-fold крос-валідація)")
        comparison_data = []
        for name, res in results.items():
            comparison_data.append({
                "Модель": ("⭐ " if name == best_name else "") + name,
                "CV F1 (середнє)": f"{res['cv_f1_mean']:.4f}",
                "CV F1 (±std)": f"±{res['cv_f1_std']:.4f}",
                "CV ROC-AUC": f"{res['cv_roc_mean']:.4f}",
                "CV Accuracy": f"{res['cv_acc_mean']:.4f}",
                "Train Precision": f"{res['train_precision']:.4f}",
                "Train Recall": f"{res['train_recall']:.4f}",
            })
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

        # Деталі найкращої моделі
        st.markdown(f"#### 🏆 Фінальна модель: {best_name}")
        best_res = results[best_name]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-score", f"{best_res['cv_f1_mean']:.3f}")
        c2.metric("ROC-AUC", f"{best_res['cv_roc_mean']:.3f}")
        c3.metric("Accuracy", f"{best_res['cv_acc_mean']:.3f}")
        c4.metric("Recall", f"{best_res['train_recall']:.3f}")

        # Confusion matrix
        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("**Матриця плутанини (train):**")
            cm = np.array(best_res["confusion_matrix"])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Прогноз: Немає", "Прогноз: Є"],
                y=["Факт: Немає", "Факт: Є"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            ))
            fig_cm.update_layout(height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_fi:
            st.markdown("**Важливість ознак:**")
            best_model_obj = best_res["model"]
            sel_feat = st.session_state.selected_features
            fi_df = get_feature_importance(best_model_obj, sel_feat)
            if not fi_df.empty:
                fig_fi = px.bar(
                    fi_df.head(12), x="importance", y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Blues",
                    labels={"feature": "Ознака", "importance": "Важливість"},
                    height=280,
                )
                fig_fi.update_layout(margin=dict(t=10, b=10), showlegend=False, coloraxis_showscale=False)
                fig_fi.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_fi, use_container_width=True)

        # Відібрані ознаки
        with st.expander("Відібрані ознаки для навчання"):
            st.write(f"Кількість ознак: **{len(sel_feat)}**")
            st.write(", ".join(sel_feat))


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 3 — ПРОГНОЗ ОПАДІВ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔮 Блок 3: Прогноз опадів</div>', unsafe_allow_html=True)

if st.session_state.best_model is None:
    st.warning("Спочатку навчіть модель у Блоці 2.")
else:
    with st.expander("Налаштування прогнозу", expanded=True):
        col1, col2 = st.columns([1, 2])

        with col1:
            forecast_days = st.slider("Кількість днів прогнозу", min_value=1, max_value=14, value=7)

        with col2:
            st.info(f"Прогноз для: **{st.session_state.location_info}**\n\n"
                    f"Використовується модель: **{st.session_state.best_name}**")

        forecast_btn = st.button("Отримати прогноз", type="primary", use_container_width=False)

    if forecast_btn:
        with st.spinner("Завантаження прогнозних даних та розрахунок..."):
            try:
                forecast_raw = fetch_forecast_data(
                    st.session_state.lat,
                    st.session_state.lon,
                    days=forecast_days,
                )
                history_df = st.session_state.df_history

                X_forecast = prepare_forecast_features(
                    forecast_raw,
                    history_df,
                    st.session_state.feature_cols,
                    st.session_state.selected_features,
                )

                pred_df = predict(
                    st.session_state.best_model,
                    X_forecast,
                    forecast_raw,
                )

                st.session_state.forecast_df = forecast_raw
                st.session_state.prediction_result = pred_df
                st.success("Прогноз готовий!")

            except Exception as e:
                st.error(f"Помилка прогнозу: {e}")
                st.stop()

    if st.session_state.prediction_result is not None:
        pred_df = st.session_state.prediction_result

        st.markdown("###Прогноз на найближчі дні")

        # Картки прогнозу
        cols = st.columns(min(len(pred_df), 7))
        for i, (_, row) in enumerate(pred_df.iterrows()):
            if i >= 7:
                break
            with cols[i % 7]:
                dt = pd.to_datetime(row["date"])
                day_label = dt.strftime("%d.%m")
                weekday = dt.strftime("%a")

                rain_prob = row.get("rain_probability", 50)
                is_rain = row["rain_predicted"] == 1

                card_class = "rain-yes" if is_rain else "rain-no"
                icon = "🌧️" if is_rain else "☀️"
                label = "Є опади" if is_rain else "Без опадів"

                temp_max = row.get("temperature_2m_max", None)
                temp_min = row.get("temperature_2m_min", None)
                temp_str = ""
                if pd.notna(temp_max) and pd.notna(temp_min):
                    temp_str = f"<br><small>🌡️ {temp_min:.0f}°...{temp_max:.0f}°C</small>"

                wind = row.get("windspeed_10m_max", None)
                wind_str = ""
                if pd.notna(wind):
                    wind_str = f"<br><small>💨 {wind:.0f} км/год</small>"

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-size:0.8rem; color:#666">{weekday}</div>
                    <div style="font-size:1rem; font-weight:700">{day_label}</div>
                    <div style="font-size:1.5rem">{icon}</div>
                    <div style="font-size:0.85rem">{label}</div>
                    <div style="font-size:0.9rem; font-weight:700">{rain_prob:.0f}%</div>
                    {temp_str}{wind_str}
                </div>
                """, unsafe_allow_html=True)

        # Якщо більше 7 днів — показуємо таблицю
        if len(pred_df) > 7:
            st.markdown("**Повний прогноз:**")

        # Зведена таблиця
        st.markdown("###Зведена таблиця прогнозу")
        display_cols = {
            "date": "Дата",
            "forecast_label": "Прогноз",
            "rain_probability": "Ймовірність опадів (%)",
            "temperature_2m_max": "Темп. макс. (°C)",
            "temperature_2m_min": "Темп. мін. (°C)",
            "temperature_2m_mean": "Темп. середня (°C)",
            "windspeed_10m_max": "Вітер макс. (км/год)",
            "precipitation_probability_mean": "Ймов. опадів Open-Meteo (%)",
        }
        avail = {k: v for k, v in display_cols.items() if k in pred_df.columns}
        show_df = pred_df[list(avail.keys())].rename(columns=avail).copy()
        show_df["Дата"] = pd.to_datetime(show_df["Дата"]).dt.strftime("%d.%m.%Y")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Графік ймовірностей
        st.markdown("###Графік ймовірності опадів")
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=pred_df["date"],
            y=pred_df["rain_probability"],
            marker_color=[
                "#1a73e8" if v == 1 else "#fbbc04"
                for v in pred_df["rain_predicted"]
            ],
            name="Ймовірність опадів",
            text=[f"{v:.0f}%" for v in pred_df["rain_probability"]],
            textposition="outside",
        ))
        fig_prob.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Поріг 50%")
        fig_prob.update_layout(
            xaxis_title="Дата",
            yaxis_title="Ймовірність опадів (%)",
            yaxis_range=[0, 115],
            height=320,
            margin=dict(t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Графік температури (якщо є)
        if "temperature_2m_max" in pred_df.columns and "temperature_2m_min" in pred_df.columns:
            st.markdown("### Температура")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=pred_df["date"], y=pred_df["temperature_2m_max"],
                name="Макс.", line=dict(color="#e53935", width=2),
                fill=None,
            ))
            fig_temp.add_trace(go.Scatter(
                x=pred_df["date"], y=pred_df["temperature_2m_min"],
                name="Мін.", line=dict(color="#1e88e5", width=2),
                fill="tonexty", fillcolor="rgba(30,136,229,0.1)",
            ))
            if "temperature_2m_mean" in pred_df.columns:
                fig_temp.add_trace(go.Scatter(
                    x=pred_df["date"], y=pred_df["temperature_2m_mean"],
                    name="Сер.", line=dict(color="#43a047", width=2, dash="dot"),
                ))
            fig_temp.update_layout(
                xaxis_title="Дата",
                yaxis_title="Температура (°C)",
                height=280,
                margin=dict(t=10, b=40),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_temp, use_container_width=True)

# ─── Підвал ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Дані: [Open-Meteo](https://open-meteo.com/) · Моделі: scikit-learn · Побудовано на Streamlit")
