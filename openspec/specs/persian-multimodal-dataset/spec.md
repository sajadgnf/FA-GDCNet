# persian-multimodal-dataset

## Requirements

### Requirement: Instagram Scraping Pipeline
The system SHALL provide a configurable script that scrapes public Instagram posts containing Persian captions, downloads the associated images, and stores text–image pairs to disk.

#### Scenario: Successful scrape under rate limits
- **WHEN** the scraper is invoked with a list of hashtags and a `--max-count` parameter
- **THEN** it SHALL fetch only public posts, persist `(post_id, caption, image_path)` rows to `datasets/raw/<hashtag>.jsonl`, and respect a configurable request-delay parameter to avoid rate-limit bans

#### Scenario: Idempotent re-runs
- **WHEN** the scraper is invoked a second time with overlapping hashtags
- **THEN** it SHALL skip posts whose `post_id` already exists in the target JSONL and SHALL resume from where the previous run left off

#### Scenario: TOS compliance recording
- **WHEN** the scraper persists data
- **THEN** it SHALL store only the image file locally and SHALL NOT redistribute the original image; the corresponding documentation in `README.md` SHALL state that only embeddings and labels may be shared externally

### Requirement: Persian Text Preprocessing
The system SHALL preprocess raw Persian captions to remove noise and standardize characters before labeling and inference.

#### Scenario: Character normalization
- **WHEN** a raw caption is loaded
- **THEN** the preprocessor SHALL normalize Persian characters (e.g. Arabic `ي → ی`, `ك → ک`), unify Arabic and Persian digits to a single form, remove URLs, remove `@mentions`, and collapse repeated whitespace

#### Scenario: Language filter
- **WHEN** a caption contains less than 50 percent Persian characters after normalization
- **THEN** the post SHALL be excluded from the labeling pool with a recorded reason `non_persian`

### Requirement: Five-Class Manual Labeling Tool
The system SHALL ship a command-line annotation tool that presents each post to a labeler and records one of five labels `{positive, negative, neutral, positive_sarcasm, negative_sarcasm}`.

#### Scenario: Labeling a sample
- **WHEN** the labeler is shown a `(caption, image)` pair
- **THEN** the tool SHALL accept a single keystroke `1`–`5` (mapped to the five classes) and persist a record `{post_id, caption, image_path, label, annotator_id, timestamp}` to `datasets/persian_multimodal_irony.jsonl`

#### Scenario: Inter-annotator agreement
- **WHEN** at least two annotators have labeled the same post
- **THEN** the tool SHALL compute Cohen's kappa over the overlapping subset and write the result to `reports/iaa.md` together with the number of overlapping samples

#### Scenario: Skip and undo
- **WHEN** the labeler presses `s` (skip) or `u` (undo) during annotation
- **THEN** the tool SHALL respectively skip the current sample without writing a record, or remove the last written record for that annotator from the JSONL

### Requirement: Dataset Storage Format
The system SHALL store the labeled dataset in a documented JSONL schema that the inference and evaluation pipelines can consume directly.

#### Scenario: Reading the dataset
- **WHEN** any downstream component opens `datasets/persian_multimodal_irony.jsonl`
- **THEN** each line SHALL be a JSON object containing at minimum the fields `{post_id, caption, image_path, label, annotators, kappa}` where `label` is one of the five class strings

#### Scenario: Schema validation
- **WHEN** the dataset loader reads a line that is missing any required field or contains an unknown label
- **THEN** the loader SHALL raise a clear `DatasetSchemaError` naming the offending line number and field
