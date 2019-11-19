import lightgbm as lgb

from src.models.lightgbm import LightGBM


def test_basic():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    data, target = load_boston(True)
    config = {"model": {}}
    config["model"]["model_params"] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "seed": 71,
        "verbose": -1
    }
    config["model"]["train_params"] = {"num_boost_round": 10}
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.2,
                                                        random_state=42)
    model = LightGBM()
    best_model, best_score = model.fit(x_train, y_train,
                                       x_test, y_test,
                                       config)
    assert isinstance(best_model, lgb.Booster)
    assert isinstance(best_score, dict)
