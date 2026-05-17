# Spec Delta: evaluation-benchmark

## ADDED Requirements

### Requirement: Accuracy and F1 Reporting
The system SHALL evaluate the trained pipeline and report Accuracy, Macro-F1, and per-class F1 over the labeled dataset using 5-fold stratified cross-validation.

#### Scenario: Producing the metrics report
- **WHEN** `eval/metrics.py` is invoked with the labeled dataset and a trained classifier
- **THEN** it SHALL emit `reports/metrics.csv` with one row per fold (containing Accuracy, Macro-F1, and per-class F1 columns) and a final row containing the `mean ± std` summary across folds

#### Scenario: Meeting the accuracy hypothesis
- **WHEN** Accuracy is reported in `reports/metrics.csv`
- **THEN** the report SHALL include a derived column or footer that explicitly states whether mean Accuracy on the sarcasm classes (`positive_sarcasm`, `negative_sarcasm`) exceeds the 70 percent hypothesis threshold

### Requirement: Latency and Memory Profiling
The system SHALL measure per-sample inference latency and peak memory usage of the inference pipeline on the active backend.

#### Scenario: Profiling on a CUDA backend
- **WHEN** evaluation runs on a CUDA-capable backend
- **THEN** the profiler SHALL record `torch.cuda.max_memory_allocated()` and the median wall-clock latency over at least 100 samples, and SHALL persist them to `reports/profile.json`

#### Scenario: Profiling on a CPU-only backend
- **WHEN** evaluation runs on a CPU-only backend
- **THEN** the profiler SHALL record peak RSS via `psutil.Process().memory_info().rss` and the median wall-clock latency over at least 100 samples, and SHALL persist them to `reports/profile.json`

#### Scenario: VRAM hypothesis check
- **WHEN** the profiler has recorded peak memory
- **THEN** `reports/profile.json` SHALL include a boolean field `under_1gib_budget` that is `true` only if peak memory remained at or below 1 GiB during the entire run

### Requirement: Ablation Study
The system SHALL perform an ablation study quantifying the contribution of each GDRM signal (Dsem, Dsen, Fvt) to final accuracy.

#### Scenario: Single-signal and pairwise runs
- **WHEN** the ablation script is invoked
- **THEN** the system SHALL re-fit the lightweight classifier on each of the subsets `{Dsem}`, `{Dsen}`, `{Fvt}`, `{Dsem, Dsen}`, `{Dsem, Fvt}`, `{Dsen, Fvt}`, and `{Dsem, Dsen, Fvt}` using the same 5-fold splits, and SHALL record Accuracy and Macro-F1 for each configuration

#### Scenario: Reporting ablation results
- **WHEN** all ablation configurations have completed
- **THEN** the system SHALL emit `reports/ablation.csv` and a bar chart `reports/ablation.png` comparing Macro-F1 across configurations

### Requirement: Unimodal Baseline Comparison
The system SHALL compare the multimodal pipeline against a text-only baseline that uses only the frozen ParsBERT polarity classifier on the caption.

#### Scenario: Baseline run
- **WHEN** the baseline evaluation is invoked
- **THEN** the system SHALL compute Accuracy and Macro-F1 of ParsBERT applied to captions on the same 5-fold splits as the multimodal pipeline, and SHALL persist results to `reports/baseline.csv`

#### Scenario: Improvement margin check
- **WHEN** both `reports/metrics.csv` (multimodal) and `reports/baseline.csv` (unimodal) exist
- **THEN** the system SHALL compute `Δ = sarcasm_F1_multimodal − sarcasm_F1_unimodal` (sarcasm F1 = macro-average of `positive_sarcasm` and `negative_sarcasm` F1) and SHALL flag in `reports/REPORT.md` whether `Δ ≥ 10` percentage points
