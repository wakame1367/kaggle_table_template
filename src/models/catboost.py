from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .base import BaseModel


class CatBoost(BaseModel):
    """
    References:
        https://www.kaggle.com/hidehisaarai1213/dsb2019-baseline/
    """
    def fit(self, x_train, y_train, x_valid, y_valid, config):
        cat_train = Pool(x_train, label=y_train)
        cat_valid = Pool(x_valid, label=y_valid)
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        if mode == "regression":
            model = CatBoostRegressor(**model_params)
        else:
            model = CatBoostClassifier(**model_params)
        model.fit(
            cat_train,
            eval_set=cat_valid,
            use_best_model=True,
            verbose=True
        )
        best_score = model.best_score_
        return model, best_score
    
    def get_best_iteration(self, model):
        return model.best_iteration_
    
    def predict(self, model, features):
        return model.predict_proba(features)[:, 1]

    def get_feature_importance(self, model):
        return model.feature_importances_
