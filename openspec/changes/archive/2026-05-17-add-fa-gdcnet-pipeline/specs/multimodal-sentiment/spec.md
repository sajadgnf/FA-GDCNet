# Spec Delta: multimodal-sentiment

## ADDED Requirements

### Requirement: Training-Free Multimodal Inference Pipeline
The system SHALL provide an end-to-end inference pipeline that accepts a (Persian caption, image) pair and returns one of five sentiment labels `{positive, negative, neutral, positive_sarcasm, negative_sarcasm}` without fine-tuning any transformer backbone.

#### Scenario: Inference on a coherent positive post
- **WHEN** the user submits a Persian caption with a matching positive image (e.g., "تولدت مبارک" with a photo of a happy birthday gathering)
- **THEN** the pipeline SHALL return the label `positive` together with a confidence score in `[0, 1]` and the full discrepancy vector

#### Scenario: Inference on a sarcastic post with contradicting modalities
- **WHEN** the caption expresses positive sentiment but the image shows a clearly negative scene (e.g., "چه روز عالی‌ای" with a flooded street)
- **THEN** the pipeline SHALL return either `positive_sarcasm` or `negative_sarcasm` and include the discrepancy vector in the response

#### Scenario: VRAM budget compliance
- **WHEN** the pipeline is loaded and runs a single inference step on any backend
- **THEN** peak VRAM (or peak RSS on CPU) SHALL remain at or below 1 GiB without an out-of-memory error

#### Scenario: Frozen backbone guarantee
- **WHEN** the inference pipeline is initialized
- **THEN** every transformer backbone (SmolVLM-256M, M-CLIP-ViT-B-32, ParsBERT) SHALL be set to `eval()` mode with `requires_grad=False` for all parameters

### Requirement: Generative Discrepancy Representation Module
The system SHALL expose a `GDRM` component that, given a (text, image) pair, computes three discrepancy signals and packs them into a feature vector consumed by the downstream classifier.

#### Scenario: Computing semantic discrepancy Dsem
- **WHEN** GDRM receives caption `T` and generated description `T̂`
- **THEN** it SHALL compute `Dsem = 1 - cosine_similarity(mCLIP_text(T), mCLIP_text(T̂))`

#### Scenario: Computing sentiment discrepancy Dsen
- **WHEN** GDRM receives caption `T` and generated description `T̂`
- **THEN** it SHALL compute `Dsen` as the absolute difference between the polarity probability vectors produced by the frozen ParsBERT classifier for `T` and for `T̂`

#### Scenario: Computing visual-textual fidelity Fvt
- **WHEN** GDRM receives image `I` and generated description `T̂`
- **THEN** it SHALL compute `Fvt = cosine_similarity(mCLIP_image(I), mCLIP_text(T̂))`

#### Scenario: Producing the discrepancy feature vector
- **WHEN** GDRM has computed Dsem, Dsen, and Fvt
- **THEN** it SHALL return a feature vector containing at minimum `{Dsem, Dsen, Fvt, cos(mCLIP_text(T), mCLIP_image(I)), polarity(T), polarity(T̂)}` suitable for direct consumption by the lightweight classifier

### Requirement: Lightweight Discrepancy Classifier
The system SHALL train a lightweight `scikit-learn` classifier on the GDRM feature vector to produce the 5-class label.

#### Scenario: Fitting the classifier
- **WHEN** the labeled dataset is available and the GDRM has produced feature vectors for every sample
- **THEN** the classifier SHALL be fit as `LogisticRegression(multi_class='multinomial', class_weight='balanced', penalty='l2')` using 5-fold stratified cross-validation, and the final model SHALL be persisted to `artifacts/clf.joblib`

#### Scenario: Reloading a trained classifier
- **WHEN** a trained classifier checkpoint exists at `artifacts/clf.joblib`
- **THEN** the inference pipeline SHALL load it on startup without retraining and serve predictions immediately

#### Scenario: Fallback classifier
- **WHEN** Logistic Regression mean cross-validated Macro-F1 is below 0.40
- **THEN** the training procedure SHALL automatically retry with `LinearSVC(class_weight='balanced')` and select whichever model yields higher mean Macro-F1

### Requirement: Hallucination Guard via Fvt Threshold
The system SHALL flag predictions whose `Fvt` falls below a configurable threshold so that generative hallucinations by SmolVLM do not silently corrupt downstream decisions.

#### Scenario: Low-fidelity rejection flag
- **WHEN** `Fvt < τ` for the current sample (default `τ = 0.2`, configurable via env or CLI flag)
- **THEN** the response SHALL include `low_fidelity: true` and the prediction SHALL still be returned but marked as low-confidence

#### Scenario: Fidelity reporting
- **WHEN** an evaluation run completes
- **THEN** the evaluation report SHALL include the rate of `low_fidelity=true` predictions across the dataset
