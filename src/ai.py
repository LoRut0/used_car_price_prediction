"""
Ядро интеллектуальной системы для прогнозирования стоимости подержанных автомобилей.

Что умеет:
1. Загружает датасет из CSV.
2. Делит данные на обучающую и тестовую выборки.
3. Автоматически обрабатывает числовые и категориальные признаки.
4. Обучает несколько моделей.
5. Сравнивает качество моделей.
6. Сохраняет лучшую модель в файл.
7. Позволяет прогнозировать цену нового автомобиля.

Ожидаемый целевой столбец: price_usd
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "price_usd"
FEATURE_PREFIX = "feature_"
FEATURES_COUNT_COLUMN = "features_count"
UP_COUNTER_COLUMN = "up_counter"
DURATION_LISTED_COLUMN = "duration_listed"


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    mae: float
    rmse: float
    r2: float


class UsedCarPriceIntelligentSystem:
    """
    Основной класс интеллектуальной системы.

    Пример использования:
        system = UsedCarPriceIntelligentSystem()
        system.train("cars.csv")
        system.save("models/car_price_model.joblib")

        price = system.predict({
            "manufacturer_name": "Subaru",
            "model_name": "Outback",
            "transmission": "automatic",
            "color": "silver",
            "odometer_value": 190000,
            "year_produced": 2010,
            "engine_fuel": "gasoline",
            "engine_has_gas": False,
            "engine_type": "gasoline",
            "engine_capacity": 2.5,
            "body_type": "universal",
            "has_warranty": False,
            "state": "owned",
            "drivetrain": "all",
            "is_exchangeable": False,
            "location_region": "Минская обл.",
            "number_of_photos": 9,
            "features_count": 7,
        })
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.best_model: Pipeline | None = None
        self.best_model_name: str | None = None
        self.feature_columns: list[str] | None = None
        self.results: list[ModelResult] = []

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        df = pd.read_csv(path)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"В датасете должен быть столбец {TARGET_COLUMN!r}")

        return df

    def train(self, csv_path: str | Path, test_size: float = 0.2) -> list[ModelResult]:
        df = self.load_data(csv_path)
        df = self._basic_cleaning(df)
        df = self._add_features_count(df)

        sample_weights = self._build_sample_weights(df)

        X = df.drop(columns=[TARGET_COLUMN])
        X = self._drop_unused_columns(X)
        X = self._prepare_features_for_model(X)
        y = df[TARGET_COLUMN]

        self.feature_columns = list(X.columns)

        X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(
            X,
            y,
            sample_weights,
            test_size=test_size,
            random_state=self.random_state,
        )

        preprocessor = self._build_preprocessor(X_train)
        models = self._build_models()

        self.results = []

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train, model__sample_weight=weights_train)
            predictions = pipeline.predict(X_test)

            result = ModelResult(
                name=model_name,
                pipeline=pipeline,
                mae=mean_absolute_error(y_test, predictions),
                rmse=np.sqrt(mean_squared_error(y_test, predictions)),
                r2=r2_score(y_test, predictions),
            )

            self.results.append(result)

        best_result = min(self.results, key=lambda item: item.mae)
        self.best_model = best_result.pipeline
        self.best_model_name = best_result.name

        return self.results

    def predict(self, car_data: dict[str, Any]) -> float:
        if self.best_model is None:
            raise RuntimeError("Модель ещё не обучена или не загружена")

        if self.feature_columns is None:
            raise RuntimeError("Неизвестен список признаков модели")

        row = pd.DataFrame([car_data])
        row = self._add_features_count(row)
        row = self._drop_unused_columns(row)

        missing_columns = set(self.feature_columns) - set(row.columns)
        if missing_columns:
            raise ValueError(f"Не хватает признаков: {sorted(missing_columns)}")

        row = row[self.feature_columns]
        row = self._prepare_features_for_model(row)
        prediction = self.best_model.predict(row)[0]

        return float(prediction)

    def save(self, model_path: str | Path) -> None:
        if self.best_model is None:
            raise RuntimeError("Нечего сохранять: модель ещё не обучена")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_columns": self.feature_columns,
        }

        joblib.dump(payload, path)

    def load(self, model_path: str | Path) -> None:
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        payload = joblib.load(path)

        self.best_model = payload["model"]
        self.best_model_name = payload["model_name"]
        self.feature_columns = payload["feature_columns"]

    def print_report(self) -> None:
        if not self.results:
            print("Нет результатов обучения")
            return

        print("\nРезультаты сравнения моделей:")
        print("-" * 72)
        print(f"{'Модель':<28} {'MAE':>12} {'RMSE':>12} {'R2':>12}")
        print("-" * 72)

        for result in sorted(self.results, key=lambda item: item.mae):
            print(
                f"{result.name:<28} "
                f"{result.mae:>12.2f} "
                f"{result.rmse:>12.2f} "
                f"{result.r2:>12.4f}"
            )

        print("-" * 72)
        print(f"Лучшая модель: {self.best_model_name}")

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.drop_duplicates()
        df = df.dropna(subset=[TARGET_COLUMN])

        # Цена не может быть отрицательной или нулевой.
        df = df[df[TARGET_COLUMN] > 0]

        # Слишком большие выбросы можно убрать по верхнему 99-му перцентилю.
        upper_price_limit = df[TARGET_COLUMN].quantile(0.99)
        df = df[df[TARGET_COLUMN] <= upper_price_limit]

        return df

    def _add_features_count(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        feature_columns = [
            column for column in df.columns if column.startswith(FEATURE_PREFIX)
        ]

        if feature_columns:
            df[FEATURES_COUNT_COLUMN] = sum(
                self._bool_like_to_int(df[column]) for column in feature_columns
            )

        return df

    def _bool_like_to_int(self, values: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(int)

        if pd.api.types.is_numeric_dtype(values):
            return values.fillna(0).astype(int).clip(lower=0, upper=1)

        true_values = {"true", "1", "yes", "y", "да", "истина"}
        return (
            values.astype("object")
            .fillna("false")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(true_values)
            .astype(int)
        )

    def _build_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        duration_listed не используется как входной признак, потому что пользователь
        вводит данные ещё не опубликованного автомобиля.

        Вместо этого столбец используется как вес обучающих объектов:
        чем дольше объявление висело на сайте, тем меньше оно влияет на модель.
        """
        if DURATION_LISTED_COLUMN not in df.columns:
            return np.ones(len(df))

        duration = pd.to_numeric(df[DURATION_LISTED_COLUMN], errors="coerce")
        duration = duration.fillna(duration.median()).clip(lower=0)

        typical_duration = max(float(duration.median()), 1.0)
        weights = 1.0 / (1.0 + duration / typical_duration)

        weights = weights / weights.mean()
        weights = weights.clip(lower=0.25, upper=2.0)

        return weights.to_numpy()

    def _drop_unused_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        columns_to_drop = [
            column
            for column in X.columns
            if column.startswith(FEATURE_PREFIX)
            or column in {UP_COUNTER_COLUMN, DURATION_LISTED_COLUMN}
        ]

        return X.drop(columns=columns_to_drop, errors="ignore")

    def _prepare_features_for_model(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        categorical_features = X.select_dtypes(
            include=["object", "str", "string", "bool", "category"],
        ).columns.tolist()

        for column in categorical_features:
            X[column] = X[column].astype("object").fillna("missing").astype(str)

        return X

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(
            include=["number"],
            exclude=["bool"],
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "str", "string", "bool", "category"],
        ).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical", categorical_transformer, categorical_features),
            ]
        )

    def _build_models(self) -> dict[str, Any]:
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=self.random_state,
            ),
        }


if __name__ == "__main__":
    system = UsedCarPriceIntelligentSystem()

    results = system.train("data/cars.csv")
    system.print_report()
    system.save("models/used_car_price_model.joblib")
