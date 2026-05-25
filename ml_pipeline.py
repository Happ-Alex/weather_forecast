"""
ml_pipeline.py — повний ML-пайплайн для прогнозування опадів
Включає: підготовку ознак, відбір ознак, навчання моделей, оцінку, прогноз
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings("ignore")


# ─── Ознаки, які є витоком даних ──────────────────────────────────────────────
LEAKAGE_COLS = {"precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формує ознаки для навчання з метеоданих:
    - часові ознаки (місяць, день року, сезон)
    - rolling-статистики (лаги та ковзне середнє)
    - різниці та похідні
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Часові ознаки
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
    })
    # Циклічне кодування місяця
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Числові стовпці (без витоку)
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in LEAKAGE_COLS and c != "rain_flag"]

    # Лагові ознаки (1, 2, 3 дні)
    for col in ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max",
                "sunshine_duration", "shortwave_radiation_sum", "weathercode"]:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Ковзні середні (3 та 7 днів)
    for col in ["temperature_2m_mean", "windspeed_10m_max", "shortwave_radiation_sum"]:
        if col in df.columns:
            df[f"{col}_roll3"] = df[col].shift(1).rolling(3, min_periods=1).mean()
            df[f"{col}_roll7"] = df[col].shift(1).rolling(7, min_periods=1).mean()

    # Різниця температур
    if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
        df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
        df["temp_range_lag1"] = df["temp_range"].shift(1)

    # Різниця температур між днями
    if "temperature_2m_mean" in df.columns:
        df["temp_diff_lag1"] = df["temperature_2m_mean"].diff(1).shift(1)

    return df


def prepare_train_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Підготовка X, y для навчання.
    Повертає X, y, список ознак.
    """
    df = build_features(df)

    # Цільова змінна
    df["rain_flag"] = (df["precipitation_sum"] > 0).astype(int)

    # Видаляємо стовпці з витоком і службові
    drop_cols = list(LEAKAGE_COLS) + ["date", "rain_flag"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df["rain_flag"].copy()

    # Прибираємо рядки з NaN (через лаги)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    # Заповнення залишкових NaN медіаною
    X = X.fillna(X.median(numeric_only=True))

    return X, y, feature_cols


def select_features(X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> List[str]:
    """
    Відбір ознак за mutual information + важливість RandomForest.
    """
    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # RandomForest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Рейтинг: середнє місце за двома методами
    rank_mi = mi_series.rank(ascending=False)
    rank_rf = rf_series.rank(ascending=False)
    combined = (rank_mi + rank_rf) / 2
    top_features = combined.sort_values().head(n_features).index.tolist()

    return top_features


def get_models(class_weights: Optional[Dict] = None) -> Dict[str, Any]:
    """Словник моделей для порівняння."""
    cw = class_weights or "balanced"
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight=cw, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, class_weight=cw, max_iter=1000, random_state=42
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=1.0, kernel="rbf", class_weight=cw,
                probability=True, random_state=42
            ))
        ]),
    }


def evaluate_models(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[str, Dict], str, Any, List[str]]:
    """
    Навчає і оцінює всі моделі через крос-валідацію.
    Повертає:
      results: dict з метриками кожної моделі
      best_name: назва найкращої моделі
      best_model: навчена фінальна модель
      selected_features: список відібраних ознак
    """
    # Відбір ознак
    selected_features = select_features(X, y, n_features=15)
    X_sel = X[selected_features]

    # Ваги класів
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    models = get_models(class_weight_dict)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        cv_f1 = cross_val_score(model, X_sel, y, cv=cv, scoring="f1", n_jobs=-1)
        cv_roc = cross_val_score(model, X_sel, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_acc = cross_val_score(model, X_sel, y, cv=cv, scoring="accuracy", n_jobs=-1)

        # Навчаємо на всіх даних для фінальних метрик
        model.fit(X_sel, y)
        y_pred = model.predict(X_sel)
        y_proba = model.predict_proba(X_sel)[:, 1] if hasattr(model, "predict_proba") else None

        results[name] = {
            "model": model,
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "cv_roc_mean": cv_roc.mean(),
            "cv_roc_std": cv_roc.std(),
            "cv_acc_mean": cv_acc.mean(),
            "cv_acc_std": cv_acc.std(),
            "train_accuracy": accuracy_score(y, y_pred),
            "train_f1": f1_score(y, y_pred),
            "train_precision": precision_score(y, y_pred),
            "train_recall": recall_score(y, y_pred),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }
        if y_proba is not None:
            results[name]["train_roc_auc"] = roc_auc_score(y, y_proba)

    # Вибір найкращої моделі за CV F1
    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
    best_model = results[best_name]["model"]

    return results, best_name, best_model, selected_features


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Витягує важливість ознак для інтерпретації моделі."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if clf is None:
            return pd.DataFrame()
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            return pd.DataFrame()
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def prepare_forecast_features(
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame,
    feature_cols: List[str],
    selected_features: List[str],
) -> pd.DataFrame:
    """
    Підготовка ознак для прогнозу (без цільової змінної).
    Використовує останні дні з history для лагів.
    """
    # Об'єднуємо хвіст history + forecast для коректного розрахунку лагів
    tail = history_df.tail(10).copy()

    # Додаємо порожній precipitation_sum у прогноз для сумісності функції
    fdf = forecast_df.copy()
    if "precipitation_sum" not in fdf.columns:
        fdf["precipitation_sum"] = np.nan
    if "rain_sum" not in fdf.columns:
        fdf["rain_sum"] = np.nan
    if "snowfall_sum" not in fdf.columns:
        fdf["snowfall_sum"] = np.nan
    if "precipitation_hours" not in fdf.columns:
        fdf["precipitation_hours"] = np.nan

    combined = pd.concat([tail, fdf], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    combined = build_features(combined)

    # Беремо тільки рядки прогнозу
    forecast_mask = combined["date"].isin(fdf["date"])
    X_forecast = combined[forecast_mask][selected_features].copy()
    X_forecast = X_forecast.fillna(X_forecast.median(numeric_only=True))

    # Якщо є NaN (мало history), заповнюємо глобальним медіаном
    X_forecast = X_forecast.fillna(0)

    return X_forecast


def predict(
    model: Any,
    X: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Робить прогноз і повертає DataFrame з результатами.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X), 0.5)

    result = forecast_df[["date"]].copy().reset_index(drop=True)
    result["rain_predicted"] = y_pred
    result["rain_probability"] = np.round(y_proba * 100, 1)
    result["forecast_label"] = result["rain_predicted"].map({
        1: "🌧️ Очікуються опади",
        0: "☀️ Опадів не очікується"
    })

    # Додаємо погодні показники якщо є
    for col in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "windspeed_10m_max", "precipitation_probability_mean", "weathercode"]:
        if col in forecast_df.columns:
            result[col] = forecast_df[col].values

    return result


def save_model(model: Any, selected_features: List[str], path: str = "model.joblib") -> None:
    joblib.dump({"model": model, "features": selected_features}, path)


def load_model(path: str = "model.joblib") -> Tuple[Any, List[str]]:
    obj = joblib.load(path)
    return obj["model"], obj["features"]
