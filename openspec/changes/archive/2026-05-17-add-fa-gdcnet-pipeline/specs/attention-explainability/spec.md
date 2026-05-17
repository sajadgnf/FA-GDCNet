# Spec Delta: attention-explainability

## ADDED Requirements

### Requirement: RTL-Aware Attention Rollout
The system SHALL implement an Attention Rollout algorithm modified for right-to-left languages so that token-level attention scores align with the natural Persian reading order in any visualization.

#### Scenario: Attention propagation across transformer layers
- **WHEN** the user requests an explanation for a prediction
- **THEN** the system SHALL compute the layer-wise multiplied attention matrices (with identity-residual addition) for the ParsBERT text encoder and produce a per-token attribution score for the input caption

#### Scenario: RTL token remap before rendering
- **WHEN** the attention scores are about to be rendered for a Persian sentence
- **THEN** the system SHALL apply `remap_rtl_indices` so that the rendered order matches the original right-to-left reading order, without altering the underlying numerical scores

#### Scenario: Image-side attention map
- **WHEN** the user requests an explanation for a prediction
- **THEN** the system SHALL produce a patch-level attention map from the M-CLIP image encoder that can be overlaid on the original image

### Requirement: Multimodal Heatmap Visualization
The system SHALL render textual attention heatmaps and image-region heatmaps for every prediction it explains.

#### Scenario: Text heatmap rendering
- **WHEN** an explanation is requested for a sample
- **THEN** the system SHALL emit an HTML artifact in which each Persian token is wrapped in a `<span>` whose background opacity is proportional to its RTL-remapped attention weight

#### Scenario: Image overlay rendering
- **WHEN** an explanation is requested for a sample
- **THEN** the system SHALL save a PNG overlay where the patch-level attention map is alpha-blended on top of the input image and stored at `reports/explain/<post_id>.png`

#### Scenario: Joint artifact retrieval
- **WHEN** the caller queries the explainability API for a known `post_id`
- **THEN** the system SHALL return file paths for both the HTML text heatmap and the PNG image overlay, or a clear error if either artifact is missing

### Requirement: Explainability Dashboard
The system SHALL provide a Streamlit-based dashboard that displays the input pair, the predicted label, the discrepancy vector, and both heatmaps for interactive inspection.

#### Scenario: Loading a sample for inspection
- **WHEN** the user selects a sample by `post_id` from the dashboard sidebar
- **THEN** the dashboard SHALL display the Persian caption, the image, the model prediction (with confidence), the full discrepancy vector, and both heatmaps within 2 seconds on a typical laptop CPU

#### Scenario: Low-fidelity warning surface
- **WHEN** the selected sample has `low_fidelity=true`
- **THEN** the dashboard SHALL display a visible warning banner explaining that the generated description had low fidelity to the image and that the prediction should be treated as low-confidence
