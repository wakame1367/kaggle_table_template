from catboost import CatBoostRegressor

from src.models.catboost import CatBoost


def test_basic():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    data, target = load_boston(True)
    config = {"model": {}}
    config["model"]["model_params"] = {
        "loss_function": "RMSE",
        "iterations": 10,
        "random_seed": 71,
        "verbose": -1
    }
    config["model"]["train_params"] = {"mode": "regression"}
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.2,
                                                        random_state=42)
    model = CatBoost()
    best_model, best_score = model.fit(x_train, y_train,
                                       x_test, y_test,
                                       config)
    assert isinstance(best_model, CatBoostRegressor)
    assert isinstance(best_score, dict)
