"""Tests for customer_churn_ml.training.trainer."""

from __future__ import annotations

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from customer_churn_ml.training.trainer import (
    ModelTrainer,
    TrainingConfig,
    TrainingOutcome,
    train_models,
)


def _no_save_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        model_path=tmp_path / "model.pkl",
        feature_names_path=tmp_path / "feature_names.pkl",
    )


# ---------------------------------------------------------------------------
# TrainingOutcome shape
# ---------------------------------------------------------------------------


class TestTrainingOutcome:
    def test_returns_training_outcome(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        outcome = trainer.train(X_train, y_train, X_test, y_test, save_artifacts=False)
        assert isinstance(outcome, TrainingOutcome)

    def test_outcome_has_required_attributes(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        outcome = trainer.train(X_train, y_train, X_test, y_test, save_artifacts=False)
        assert outcome.best_model_name
        assert outcome.best_model is not None
        assert isinstance(outcome.results, dict)
        assert isinstance(outcome.comparison_table, list)
        assert isinstance(outcome.feature_names, list)

    def test_feature_names_match_train_columns(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        outcome = trainer.train(X_train, y_train, X_test, y_test, save_artifacts=False)
        assert outcome.feature_names == list(X_train.columns)

    def test_results_contains_all_models(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        models = {
            "DummyA": DummyClassifier(strategy="most_frequent"),
            "DummyB": DummyClassifier(strategy="stratified"),
        }
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        outcome = trainer.train(
            X_train,
            y_train,
            X_test,
            y_test,
            models=models,
            save_artifacts=False,
        )
        assert set(outcome.results.keys()) == {"DummyA", "DummyB"}


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------


class TestBestModelSelection:
    def test_best_model_selected_by_roc_auc(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        models = {
            "Dummy": DummyClassifier(strategy="most_frequent", random_state=42),
            "LR": LogisticRegression(max_iter=200, random_state=42),
        }
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        outcome = trainer.train(
            X_train,
            y_train,
            X_test,
            y_test,
            models=models,
            save_artifacts=False,
        )
        # Best model must be one of the candidates
        assert outcome.best_model_name in {"Dummy", "LR"}
        # Best model must have the highest or equal ROC-AUC
        best_auc = outcome.results[outcome.best_model_name]["roc_auc"]
        for _name, result in outcome.results.items():
            assert best_auc >= (result["roc_auc"] or -1)

    def test_best_model_is_fitted(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        outcome = trainer.train(X_train, y_train, X_test, y_test, save_artifacts=False)
        # A fitted model must be able to predict
        preds = outcome.best_model.predict(X_test)
        assert len(preds) == len(X_test)


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


class TestArtifactSaving:
    def test_model_file_written_when_save_true(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        model_path = tmp_path / "model.pkl"
        config = TrainingConfig(
            model_path=model_path,
            feature_names_path=tmp_path / "fn.pkl",
        )
        trainer = ModelTrainer(config)
        trainer.train(X_train, y_train, X_test, y_test, save_artifacts=True)
        assert model_path.exists()

    def test_feature_names_file_written_when_save_true(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        fn_path = tmp_path / "fn.pkl"
        config = TrainingConfig(
            model_path=tmp_path / "model.pkl",
            feature_names_path=fn_path,
        )
        trainer = ModelTrainer(config)
        trainer.train(X_train, y_train, X_test, y_test, save_artifacts=True)
        assert fn_path.exists()

    def test_no_files_written_when_save_false(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        model_path = tmp_path / "model.pkl"
        fn_path = tmp_path / "fn.pkl"
        config = TrainingConfig(model_path=model_path, feature_names_path=fn_path)
        trainer = ModelTrainer(config)
        trainer.train(X_train, y_train, X_test, y_test, save_artifacts=False)
        assert not model_path.exists()
        assert not fn_path.exists()


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------


class TestTrainModels:
    def test_functional_wrapper_returns_outcome(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        config = _no_save_config(tmp_path)
        outcome = train_models(
            X_train,
            y_train,
            X_test,
            y_test,
            save_artifacts=False,
            config=config,
        )
        assert isinstance(outcome, TrainingOutcome)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_raises_when_no_models_provided(self, split, tmp_path):
        X_train, X_test, y_train, y_test = split
        config = _no_save_config(tmp_path)
        trainer = ModelTrainer(config)
        with pytest.raises((ValueError, Exception)):
            trainer.train(
                X_train,
                y_train,
                X_test,
                y_test,
                models={},
                save_artifacts=False,
            )
