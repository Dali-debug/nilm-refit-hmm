"""
End-to-end pipeline tests using synthetic data.
Covers: cleaning → states → train → infer → metrics.
"""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_synthetic_house
from src.cleaning import clean_resample
from src.states import build_state_definitions
from src.hmm_train import train_appliance_model
from src.inference_greedy import greedy_disaggregate, predict_appliance
from src.metrics import evaluate_all, mae, rmse, nrmse, ree, f1_on_off


# ── Cleaning tests ─────────────────────────────────────────────────────────────

class TestCleanResample:
    def test_no_negatives_after_clean(self, synthetic_house):
        clean, _ = clean_resample(synthetic_house)
        assert (clean.fillna(0) >= 0).all().all()

    def test_output_columns_preserved(self, synthetic_house):
        clean, _ = clean_resample(synthetic_house)
        assert set(synthetic_house.columns) == set(clean.columns)

    def test_report_stats(self, synthetic_house):
        _, stats = clean_resample(synthetic_house, report=True)
        assert stats is not None
        assert "rows_before" in stats
        assert "interpolated" in stats

    def test_negatives_removed(self):
        df = make_synthetic_house(n_minutes=120)
        df.iloc[5, 0] = -100.0  # inject negative
        clean, stats = clean_resample(df, report=True)
        assert (clean.fillna(0) >= 0).all().all()

    def test_resample_reduces_rows(self):
        """After resampling 8-second data to 1min, rows should reduce."""
        # Synthetic data is already at 1min, so rows should be roughly the same
        df = make_synthetic_house(n_minutes=600)
        clean, _ = clean_resample(df, resample_rule="1min")
        assert len(clean) <= len(df)


# ── State tests ────────────────────────────────────────────────────────────────

class TestBuildStateDefs:
    def test_two_states_kettle(self, synthetic_house):
        defs = build_state_definitions(synthetic_house["kettle"], "kettle", n_states=2)
        assert defs["n_states"] == 2
        assert len(defs["means"]) == 2
        assert len(defs["vars"]) == 2
        assert defs["labels"][0] == "OFF"

    def test_three_states_fridge(self, synthetic_house):
        defs = build_state_definitions(synthetic_house["fridge"], "fridge", n_states=3)
        assert defs["n_states"] == 3
        assert defs["means"][0] <= defs["means"][1]  # ordered by power

    def test_means_ordered(self, synthetic_house):
        defs = build_state_definitions(synthetic_house["washing_machine"], "washing_machine", n_states=4)
        means = defs["means"]
        assert all(means[i] <= means[i+1] + 1e-6 for i in range(len(means)-1))

    def test_vars_positive(self, synthetic_house):
        defs = build_state_definitions(synthetic_house["fridge"], "fridge", n_states=3)
        assert (np.array(defs["vars"]) > 0).all()


# ── HMM training tests ─────────────────────────────────────────────────────────

class TestHMMTrain:
    @pytest.fixture(scope="class")
    def trained_models(self, synthetic_houses):
        """Train simple 2-state models on 2 synthetic houses."""
        models = {}
        for app in ["kettle", "fridge"]:
            series_list = [synthetic_houses[i][app] for i in range(2)]
            models[app] = train_appliance_model(series_list, app, n_states=2, n_iter=5)
        return models

    def test_models_fitted(self, trained_models):
        for app, mdl in trained_models.items():
            assert hasattr(mdl, "means_"), f"Model for {app} not fitted."

    def test_model_has_correct_n_states(self, trained_models):
        for mdl in trained_models.values():
            assert mdl.n_components == 2

    def test_no_data_raises(self):
        from src.hmm_train import train_appliance_model
        empty = [pd.Series(dtype=float)]
        with pytest.raises(ValueError):
            train_appliance_model(empty, "kettle", n_states=2)


# ── Inference tests ────────────────────────────────────────────────────────────

class TestGreedyInference:
    @pytest.fixture(scope="class")
    def trained_models(self, synthetic_houses):
        models = {}
        for app in ["kettle", "fridge"]:
            series_list = [synthetic_houses[i][app] for i in [0, 1]]
            models[app] = train_appliance_model(series_list, app, n_states=2, n_iter=10)
        return models

    def test_output_shape(self, synthetic_houses, trained_models):
        test_house = synthetic_houses[2]
        preds = greedy_disaggregate(test_house["mains"], trained_models)
        assert len(preds) == len(test_house)

    def test_output_non_negative(self, synthetic_houses, trained_models):
        test_house = synthetic_houses[2]
        preds = greedy_disaggregate(test_house["mains"], trained_models)
        assert (preds.fillna(0) >= 0).all().all()

    def test_output_columns(self, synthetic_houses, trained_models):
        test_house = synthetic_houses[2]
        preds = greedy_disaggregate(test_house["mains"], trained_models)
        assert set(trained_models.keys()) == set(preds.columns)


# ── Metrics tests ──────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mae_perfect(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert mae(arr, arr) == pytest.approx(0.0)

    def test_rmse_perfect(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert rmse(arr, arr) == pytest.approx(0.0)

    def test_nrmse_nan_on_constant(self):
        arr = np.ones(10)
        # Range = 0 → NRMSE should be NaN
        assert np.isnan(nrmse(arr, arr * 1.1))

    def test_ree_perfect(self):
        arr = np.array([100.0, 200.0, 300.0])
        assert ree(arr, arr) == pytest.approx(0.0, abs=1e-6)

    def test_f1_perfect(self):
        arr = np.array([0.0, 1000.0, 0.0, 1000.0])
        assert f1_on_off(arr, arr, threshold=10.0) == pytest.approx(1.0)

    def test_f1_all_wrong(self):
        true = np.array([1000.0, 0.0, 1000.0, 0.0])
        pred = np.array([0.0, 1000.0, 0.0, 1000.0])
        assert f1_on_off(true, pred, threshold=10.0) == pytest.approx(0.0)

    def test_evaluate_all_returns_dataframe(self, synthetic_houses):
        # Use synthetic data as both true and pred (perfect case)
        df = synthetic_houses[0][["kettle", "fridge"]]
        result = evaluate_all(df, df, appliances=["kettle", "fridge"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "MAE" in result.columns

    def test_evaluate_all_with_house_id(self, synthetic_houses):
        df = synthetic_houses[0][["kettle"]]
        result = evaluate_all(df, df, appliances=["kettle"], house_id=99)
        assert "house" in result.columns
        assert result["house"].iloc[0] == 99
