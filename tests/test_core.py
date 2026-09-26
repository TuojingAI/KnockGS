import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from response_calibration.calibration import (parameter_errors, predict_knn, predict_nearest,
                                 ridge_fit_predict)
from response_calibration.features import extract_five_features
from response_calibration.__main__ import read_rows

ROOT = Path(__file__).resolve().parents[1] / "examples" / "ficus"


class CoreTests(unittest.TestCase):
    def test_archived_ficus_predictions(self):
        library = read_rows(ROOT / "library.csv", labeled=True)
        queries = read_rows(ROOT / "queries.csv")
        with (ROOT / "ground_truth.csv").open(newline="") as f:
            truth = {r["case_name"]: r for r in csv.DictReader(f)}
        with (ROOT / "expected_predictions.csv").open(newline="") as f:
            expected = {(r["target_case"], r["method"]): r for r in csv.DictReader(f)}
        methods = {
            "fixed_default": lambda q: (1., 1., {}),
            "response_nearest": lambda q: predict_nearest(library, q),
            "response_knn": lambda q: predict_knn(library, q, 3, "inverse"),
            "global_ridge": lambda q: ridge_fit_predict(library, q, .01),
            "local_ridge": lambda q: ridge_fit_predict(library, q, .001, 10),
        }
        self.assertEqual(len(library), 54)
        self.assertEqual(len(queries), 5)
        for q in queries:
            self.assertNotIn("E_scale", q)
            for method, predict in methods.items():
                with self.subTest(query=q["case_name"], method=method):
                    e, rho, _ = predict(q)
                    ref = expected[q["case_name"], method]
                    np.testing.assert_allclose([e, rho],
                        [float(ref["estimated_E_scale"]), float(ref["estimated_density_scale"])],
                        rtol=1e-9, atol=1e-10)
                    err = parameter_errors(truth[q["case_name"]], e, rho)[2]
                    self.assertAlmostEqual(err, float(ref["joint_error_percent"]), places=7)
            # Query labels must never influence calibration.
            changed = dict(q, E_scale=999, density_scale=999)
            np.testing.assert_equal(methods["local_ridge"](q)[:2],
                                    methods["local_ridge"](changed)[:2])

    def test_descriptor_on_known_translation(self):
        # Analytical check only; this synthetic sequence is not paper evidence.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "physical_state"
            folder.mkdir()
            initial = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
            for i in range(31):
                np.savez(folder / f"frame_{i:03d}.npz",
                         positions=initial + np.array([0., i, 0.], dtype=np.float32))
            f = extract_five_features(Path(tmp))
            self.assertEqual(f["norm_auc_rms_disp"], 465.)
            self.assertEqual(f["norm_initial_slope_rms_0_5"], 1.)
            self.assertEqual(f["norm_peak_rms_velocity_proxy"], 1.)
            self.assertEqual(f["norm_auc_rms_velocity_proxy"], 30.)
            self.assertAlmostEqual(f["early_velocity_ratio"], 1 / 6)
            (folder / "frame_030.npz").unlink()
            with self.assertRaises(RuntimeError):
                extract_five_features(Path(tmp))


if __name__ == "__main__":
    unittest.main()
