# Core code: partial release

This release contains the actual response descriptor and calibration routines
extracted from the experiment implementation, with machine-specific environment
checks and experiment orchestration removed. The numerical estimator is unchanged.
It runs on a CPU with Python 3.10+ and NumPy.

## Quick start

Run from the repository root:

```sh
python -m pip install -r requirements.txt
python -m response_calibration --library examples/ficus/library.csv --queries examples/ficus/queries.csv --output predictions.json
python -m unittest discover -s tests -v
```

The example has 54 candidate responses and five held-out Ficus queries from the
existing simulator experiments. It is real precomputed feature data, not synthetic
benchmark data. Only neutral case identifiers and selected CSV columns were
changed; numeric feature and parameter values are preserved.

`ground_truth.csv` is for evaluation only. The estimator reads only `library.csv`
and `queries.csv`; query files contain no target material labels. The tests compare
all five methods against 25 archived predictions. The expected mean joint parameter
error for local ridge on this **Ficus** fixture is approximately **0.62909%**.
This is not the five-target Pillow result shown in the main README.

## Included implementation

- `response_calibration/features.py`: the deterministic five-dimensional descriptor.
- `response_calibration/calibration.py`: candidate-only standardization, nearest response,
  response KNN, global ridge, local ridge, and parameter-error calculation.
- `response_calibration/__main__.py`: a portable CSV-to-JSON inference entry point with fixed
  local-ridge settings (`k=10`, `alpha=0.001`).
- `examples/ficus/`: small precomputed feature fixture and archived reference results.
- `tests/test_core.py`: regression against archived results and an analytical
  descriptor check using a clearly synthetic translation sequence.

## Descriptor input and convention

```python
from pathlib import Path
from response_calibration.features import extract_five_features

features = extract_five_features(Path("my_case"))
```

`my_case/physical_state/` must contain 31 zero-padded `frame_*.npz` files in time
order (initial state plus 30 response frames). Each file should store `positions`,
an `(N, 3)` array in consistent units with the same particle order and count in all
frames. For compatibility with the original export, the first NPZ array is used
when `positions` is absent. Inputs must be finite and the initial bounding box
must be nondegenerate. The descriptor uses all supplied particles and float32
position arithmetic, as in the experiment implementation.

The five features are: summed RMS displacement divided by initial bounding-box
diagonal; the frame-0-to-5 displacement slope divided by that diagonal; peak and
summed RMS frame-to-frame displacement divided by that diagonal; and the fraction
of summed frame-to-frame displacement occurring in frames 1 through 5. The
historical `auc` names denote discrete sums, and `velocity_proxy` is per-frame
displacement, not physical velocity. Use the same frame sampling and simulator
contract for the candidate library and the query. Libraries are object-specific.

## Estimation and frozen prediction

Standardization statistics use only the entire candidate library. Local ridge
selects the ten nearest standardized responses using Euclidean distance, then
fits two unweighted ridge models with an unregularized intercept. Predictions
are clipped to candidate parameter ranges. Ties use the case identifier.
The original fallback is inverse-distance KNN when there are fewer than three
neighbors or the linear solve fails; fallback and clipping are recorded in output.

The regression fixture also includes nearest-response retrieval, KNN (`k=3`,
inverse-distance weights), and global ridge (`alpha=0.01`). Those baseline settings
come from the original Ficus candidate-only cross-validation; the full tuning
pipeline is outside this partial release.

For a full A-to-B experiment, estimate scales using Probe A, freeze the resulting
values, and write them into the same simulator before executing held-out Probe B.
Probe B responses and held-out labels must not be used for estimation or tuning.

## Scope

This is a **partial core-code release**, not a complete reproduction package.
It does not include Gaussian assets, particle trajectories, the MPM simulator,
rendering, asset reconstruction, robot control, or the full experiment orchestration.
The bundled example reproduces feature-to-parameter estimates, not Probe-B
simulations, image-quality metrics, or every table in the paper. No GPU is needed
for this subset. Existing upstream projects are credited in the main README;
their source code is not bundled here.
