from ai import UsedCarPriceIntelligentSystem


def main() -> None:
    system = UsedCarPriceIntelligentSystem()
    system.load("models/used_car_price_model.joblib")

    car = {
        "manufacturer_name": "LADA",
        "model_name": "Priora",
        "transmission": "mechanical",
        "color": "silver",
        "odometer_value": 190000,
        "year_produced": 2010,
        "engine_fuel": "gasoline",
        "engine_has_gas": False,
        "engine_type": "gasoline",
        "engine_capacity": 1.6,
        "body_type": "sedan",
        "has_warranty": False,
        "state": "owned",
        "drivetrain": "front",
        "is_exchangeable": False,
        "location_region": "Минская обл.",
        "number_of_photos": 9,
        # Вместо feature_0 ... feature_9
        "features_count": 0,
    }

    predicted_price = system.predict(car)

    print(f"Используемая модель: {system.best_model_name}")
    print(f"Прогнозируемая цена автомобиля: {predicted_price:.2f} USD")


if __name__ == "__main__":
    main()
