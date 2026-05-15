from pathlib import Path
import sys
import inspect
import importlib
import json
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, field_validator


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

MODEL_PATH = ROOT_DIR / "models" / "used_car_price_model.joblib"
CATALOG_PATH = ROOT_DIR / "models" / "car_catalog.json"

FRONTEND_DIR = ROOT_DIR / "frontend"

DATASET_CANDIDATES = [
    ROOT_DIR / "data" / "cars.csv",
    ROOT_DIR / "data" / "cars_dataset.csv",
    ROOT_DIR / "data" / "used_cars.csv",
]

sys.path.insert(0, str(SRC_DIR))


class CarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manufacturer_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    transmission: str = Field(min_length=1)
    color: str = Field(min_length=1)

    odometer_value: int = Field(ge=0)
    year_produced: int = Field(ge=1900, le=2030)

    engine_fuel: str = Field(min_length=1)
    engine_has_gas: bool
    engine_type: str = Field(min_length=1)
    engine_capacity: float | None = Field(default=None, ge=0)

    body_type: str = Field(min_length=1)
    has_warranty: bool
    state: str = Field(min_length=1)
    drivetrain: str = Field(min_length=1)

    is_exchangeable: bool
    location_region: str = Field(min_length=1)

    number_of_photos: int = Field(default=0, ge=0)

    # Вместо feature_0 ... feature_9
    features_count: int = Field(default=0, ge=0, le=10)

    @field_validator(
        "manufacturer_name",
        "model_name",
        "transmission",
        "color",
        "engine_fuel",
        "engine_type",
        "body_type",
        "state",
        "drivetrain",
        "location_region",
        mode="before",
    )
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class PredictionResponse(BaseModel):
    price_usd: float
    currency: str = "USD"


class CatalogService:
    def __init__(self, catalog_path: Path, dataset_candidates: list[Path]):
        self.catalog_path = catalog_path
        self.dataset_candidates = dataset_candidates
        self.catalog: dict[str, list[str]] = {}

    def load(self) -> None:
        if self.catalog_path.exists():
            self.catalog = self._load_from_json(self.catalog_path)
            return

        dataset_path = self._find_dataset()
        if dataset_path is None:
            raise FileNotFoundError(
                "Car catalog was not found and dataset was not found. "
                f"Expected catalog: {self.catalog_path}. "
                f"Dataset candidates: {[str(path) for path in self.dataset_candidates]}"
            )

        self.catalog = self._build_from_dataset(dataset_path)
        self._save_to_json(self.catalog_path, self.catalog)

    def _load_from_json(self, path: Path) -> dict[str, list[str]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise RuntimeError("Catalog JSON must contain an object")

        catalog: dict[str, list[str]] = {}

        for manufacturer, models in data.items():
            if not isinstance(manufacturer, str):
                continue

            if not isinstance(models, list):
                continue

            clean_models = sorted(
                {str(model).strip() for model in models if str(model).strip()}
            )

            if clean_models:
                catalog[manufacturer.strip()] = clean_models

        if not catalog:
            raise RuntimeError("Catalog is empty")

        return dict(sorted(catalog.items()))

    def _find_dataset(self) -> Path | None:
        for path in self.dataset_candidates:
            if path.exists():
                return path
        return None

    def _build_from_dataset(self, dataset_path: Path) -> dict[str, list[str]]:
        df = pd.read_csv(dataset_path)

        required_columns = {"manufacturer_name", "model_name"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise RuntimeError(
                f"Dataset does not contain required columns: {sorted(missing_columns)}"
            )

        cars = df[["manufacturer_name", "model_name"]].dropna().copy()

        cars["manufacturer_name"] = cars["manufacturer_name"].astype(str).str.strip()
        cars["model_name"] = cars["model_name"].astype(str).str.strip()

        cars = cars[(cars["manufacturer_name"] != "") & (cars["model_name"] != "")]

        catalog: dict[str, list[str]] = {}

        for manufacturer, group in cars.groupby("manufacturer_name"):
            models = sorted(group["model_name"].unique().tolist())
            catalog[manufacturer] = models

        if not catalog:
            raise RuntimeError(
                "Cannot build catalog: no valid manufacturer/model pairs"
            )

        return dict(sorted(catalog.items()))

    def _save_to_json(self, path: Path, catalog: dict[str, list[str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(catalog, file, ensure_ascii=False, indent=2)

    def validate_car(self, manufacturer_name: str, model_name: str) -> None:
        if not self.catalog:
            raise HTTPException(
                status_code=503,
                detail="Car catalog is not loaded",
            )

        if manufacturer_name not in self.catalog:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Марка '{manufacturer_name}' отсутствует в каталоге",
                    "field": "manufacturer_name",
                },
            )

        allowed_models = self.catalog[manufacturer_name]

        if model_name not in allowed_models:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        f"Модель '{model_name}' не найдена "
                        f"для марки '{manufacturer_name}'"
                    ),
                    "field": "model_name",
                    "manufacturer_name": manufacturer_name,
                    "allowed_models": allowed_models,
                },
            )


class ModelService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.system: Any | None = None
        self.raw_model: Any | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Сначала пробуем загрузить твою систему из src/ai.py
        try:
            self.system = self._load_ai_system_from_src()
            return
        except Exception:
            self.system = None

        # Запасной вариант: если joblib содержит сразу pipeline/model
        self.raw_model = joblib.load(self.model_path)

    def _load_ai_system_from_src(self) -> Any:
        ai_module = importlib.import_module("ai")

        for _, cls in inspect.getmembers(ai_module, inspect.isclass):
            if cls.__module__ != ai_module.__name__:
                continue

            has_load = hasattr(cls, "load")
            has_predict = hasattr(cls, "predict") or hasattr(cls, "predict_one")

            if has_load and has_predict:
                system = cls()
                system.load(self.model_path)
                return system

        raise RuntimeError("Suitable AI system class was not found in src/ai.py")

    def predict(self, car: CarInput) -> float:
        row = car.model_dump()
        df = pd.DataFrame([row])

        if self.system is not None:
            return self._predict_with_ai_system(row, df)

        if self.raw_model is not None:
            return self._predict_with_raw_model(df)

        raise RuntimeError("Model is not loaded")

    def _predict_with_ai_system(self, row: dict, df: pd.DataFrame) -> float:
        errors = []

        for method_name in ("predict_one", "predict"):
            if not hasattr(self.system, method_name):
                continue

            method = getattr(self.system, method_name)

            for arg in (row, df):
                try:
                    result = method(arg)
                    return self._extract_price(result)
                except Exception as exc:
                    errors.append(f"{method_name}({type(arg).__name__}): {exc}")

        raise RuntimeError("Cannot make prediction. Errors: " + " | ".join(errors))

    def _predict_with_raw_model(self, df: pd.DataFrame) -> float:
        payload = self.raw_model

        if hasattr(payload, "predict"):
            result = payload.predict(df)
            return self._extract_price(result)

        if isinstance(payload, dict):
            model = (
                payload.get("pipeline")
                or payload.get("model")
                or payload.get("regressor")
            )

            preprocessor = payload.get("preprocessor") or payload.get("transformer")

            feature_columns = (
                payload.get("feature_columns")
                or payload.get("input_columns")
                or payload.get("columns")
            )

            if feature_columns is not None:
                df = df.reindex(columns=feature_columns)

            if preprocessor is not None:
                x = preprocessor.transform(df)
            else:
                x = df

            if model is None or not hasattr(model, "predict"):
                raise RuntimeError("Joblib payload does not contain a valid model")

            result = model.predict(x)
            return self._extract_price(result)

        raise RuntimeError("Unsupported model format")

    @staticmethod
    def _extract_price(result: Any) -> float:
        if isinstance(result, dict):
            for key in (
                "predicted_price_usd",
                "predicted_price",
                "price_usd",
                "prediction",
                "price",
            ):
                if key in result:
                    return round(float(result[key]), 2)

        if isinstance(result, pd.DataFrame):
            return round(float(result.iloc[0, 0]), 2)

        if isinstance(result, pd.Series):
            return round(float(result.iloc[0]), 2)

        if isinstance(result, list):
            first = result[0]
            if isinstance(first, dict):
                return ModelService._extract_price(first)
            return round(float(first), 2)

        if hasattr(result, "ravel"):
            return round(float(result.ravel()[0]), 2)

        return round(float(result), 2)


model_service = ModelService(MODEL_PATH)
catalog_service = CatalogService(CATALOG_PATH, DATASET_CANDIDATES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog_service.load()
    model_service.load()
    yield


app = FastAPI(
    title="Used Car Price Prediction API",
    description="Backend для предсказания цены подержанного автомобиля",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": model_service.system is not None
        or model_service.raw_model is not None,
        "catalog_path": str(CATALOG_PATH),
        "catalog_loaded": bool(catalog_service.catalog),
        "manufacturers_count": len(catalog_service.catalog),
        "models_count": sum(len(models) for models in catalog_service.catalog.values()),
    }


@app.get("/api/catalog")
def get_catalog():
    """
    Полный каталог:
    {
        "Audi": ["100", "A4", "A6"],
        "BMW": ["3 Series", "5 Series"]
    }
    """
    return catalog_service.catalog


@app.get("/api/catalog/manufacturers")
def get_manufacturers():
    """
    Список всех доступных марок.
    """
    return {"manufacturers": list(catalog_service.catalog.keys())}


@app.get("/api/catalog/models")
def get_models(
    manufacturer_name: str = Query(..., min_length=1),
):
    """
    Список моделей для выбранной марки.

    Пример:
    /api/catalog/models?manufacturer_name=Subaru
    """
    manufacturer_name = manufacturer_name.strip()

    if manufacturer_name not in catalog_service.catalog:
        raise HTTPException(
            status_code=400,
            detail=f"Марка '{manufacturer_name}' отсутствует в каталоге",
        )

    return {
        "manufacturer_name": manufacturer_name,
        "models": catalog_service.catalog[manufacturer_name],
    }


@app.get("/api/meta")
def get_meta():
    return {
        "transmission": ["automatic", "mechanical"],
        "engine_has_gas": [False, True],
        "has_warranty": [False, True],
        "is_exchangeable": [False, True],
        "features_count_range": {
            "min": 0,
            "max": 10,
        },
        "catalog_endpoint": "/api/catalog",
        "note": "up_counter and duration_listed are not required from user",
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_price(car: CarInput):
    catalog_service.validate_car(
        manufacturer_name=car.manufacturer_name,
        model_name=car.model_name,
    )

    try:
        predicted_price = model_service.predict(car)

        return PredictionResponse(
            price_usd=predicted_price,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )


if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )
