# Grid coverage limits tiny-object recall of a spiking detector in aerial images

Companion code and aggregate results for the study on SpikeYOLO-s and AI-TOD-v2.

## Reproduce the tables and figures

Run these commands from the repository root with Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python -m analysis.summarize
python -m analysis.make_figures
```

The summary prints the validation, held-out test, and same-weight readout tables.
The plotting script writes Figures 1–3 as PNG and PDF files under `output/figures/`.
Both scripts use the included result file and run without a GPU, dataset, or checkpoint.

- [`data/results.csv`](data/results.csv) contains per-seed validation and test metrics,
  operating points, cost estimates, and results for the signed-readout and P2 geometry
  analyses. AP and recall are given in percent. The validation recall in Table 1 uses
  the same-seed matched false-positive budget, whereas the held-out test recall uses a
  fixed score threshold of 0.25. `B_trainval` is the separately registered seed-7 test
  reference, trained on train plus validation. It is excluded from the paired comparisons.
  Energy values are operation-based estimates. The measured GPU latency includes
  preprocessing, decoding, and NMS.
- [`data/test_bootstrap.json`](data/test_bootstrap.json) contains the held-out test
  recall differences (`B - A`, in percentage points) and 95% parent-scene bootstrap
  intervals at score 0.25. Here A and B are the file's comparison labels. The intervals
  use 2,000 resamples within each training seed.

## Inspect the model and analysis code

The [model definitions](model/arms.py) implement the five study arms: B, STAL,
STAL-SD (shown as `STAL_SD` in code and data), P2, and TWIN. They use
[BICLab/SpikeYOLO at commit `dc51f35d0b6e970cd157080c98157a41af4392f5`](https://github.com/BICLab/SpikeYOLO/tree/dc51f35d0b6e970cd157080c98157a41af4392f5)
with the upstream dependencies and the official 23M, T=1, D=4 COCO initialization
checkpoint linked from that project. Use the upstream source on `PYTHONPATH`
when importing `model.arms`. Model construction requires CUDA in the pinned
upstream code. For example, call `build_model("STAL_SD", weight=checkpoint_path, device="cuda", seed=0)`
from `model.arms` with the official checkpoint. The initialization checkpoint used
in the study has SHA-256 `3b4083707de523f101afc3cbfbb208d1b4c19462a0ff16cd30509ccee2b69881`.
Use separate Python processes for TWIN and the spiking arms because the firing
function is switched at process scope.

[`analysis/coverage.py`](analysis/coverage.py) computes strict grid coverage and
the non-negative LTRB IoU ceiling. [`analysis/scene_bootstrap.py`](analysis/scene_bootstrap.py)
can recompute one pair of test intervals if supplied with the public COCO annotations
and the two corresponding evaluator status files:

```sh
python -m analysis.scene_bootstrap aitodv2_test.json baseline_per_gt_status.json variant_per_gt_status.json
```

Each status file is a JSON object with a `"0.25"` map from annotation IDs to
`[matched_at_iou_0_50, matched_at_iou_0_75]` flags, as well as the `"all"` map
from the evaluator's multi-threshold export.

This package reproduces the reported tables and figures from aggregate results.
The [AI-TOD-v2 data and annotations](https://github.com/Chasel-Tsui/AI-TOD-v2)
are available from the dataset authors. The full training pipeline, trained study
checkpoints, per-object match files, and images are outside the scope of this
release.

## Attribution

The model and loss adapter derives from BICLab SpikeYOLO and the Ultralytics code
in that repository. The adapter preserves the study's implementation of the
candidate-only STAL rule from [YOLO26 v1](https://arxiv.org/abs/2606.03748v1)
and the signed regression support motivated by
[DSDL](https://doi.org/10.1016/j.autcon.2026.106929).
The code in this repository is released under the [GNU AGPL v3.0](LICENSE).
