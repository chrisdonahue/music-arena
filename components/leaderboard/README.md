# Music Arena Leaderboard

The leaderboard is computed transparently from the public [Music Arena Dataset](https://huggingface.co/datasets/music-arena/music-arena-dataset) on HuggingFace.

## Scoring Methodology

- **Arena Score**: [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) via L2-regularized logistic regression. Ties are split as half-win / half-loss for each side. Votes with `BOTH_BAD` preference are excluded.
- **95% CI**: Bootstrap resampling (1,000 iterations)
- **Generation Speed (RTF)**: Median Real-Time Factor (audio duration / generation time), normalized to A6000 GPU for open-weights models.
- **Threshold**: Only models with 30+ votes are shown.

For the full scoring implementation, see [`ma_leaderboard/scoring.py`](ma_leaderboard/scoring.py).

## Setup (Docker)

The leaderboard runs inside Docker, like all other Music Arena components.

```bash
# Build and run any leaderboard command:
ma-comp leaderboard <command> [options]
```

## Reproduce the Leaderboard (Anyone)

No credentials required — uses only public HuggingFace data:

```bash
# Generate leaderboard from public HuggingFace dataset
ma-comp leaderboard leaderboard --output-dir results

# View the generated files
ls results/leaderboards/   # TSV tables
ls results/plots/           # PNG scatter plots
```

## Monthly Data Pipeline (Maintainers)

The full pipeline can be run with a single command:

```bash
bash components/leaderboard/monthly_update.sh
```

This will download new data, preprocess, push to HuggingFace, generate the leaderboard, and update the frontend.

### Secrets

GCP authentication uses the same `GCP_BUCKET_SERVICE_ACCOUNT` service account JSON as the gateway component. Bucket names are managed via `music_arena.secret`. On first run, you will be prompted to enter each secret interactively. Values are cached locally in `cache/secrets/` and never committed.

Required secrets:
- `GCP_BUCKET_SERVICE_ACCOUNT` — Service account JSON (shared with gateway; provide the JSON file path when prompted)
- `METADATA_BUCKET` — GCP bucket name for battle metadata
- `AUDIO_BUCKET` — GCP bucket name for audio files

### Individual Pipeline Steps

#### Step 1: Download new battle data from GCP

```bash
ma-comp leaderboard download
```

- **Start date**: Auto-detected from existing data. If no data exists, defaults to 2025-07-28 (launch date).
- **End date**: Auto-detected as end of previous month. Override with `--start` / `--end` if needed.
- Only battles with valid models (in `MODELS_METADATA`) are downloaded. Test/unknown models are skipped.
- Already downloaded files are automatically skipped (incremental).

#### Step 2: Preprocess into HuggingFace dataset format

```bash
ma-comp leaderboard preprocess
```

#### Step 3: Push to HuggingFace dataset

Preprocessed files are written to `cache/bucket/dataset/` (i.e. `CACHE_DIR/bucket/dataset/`).

```bash
git clone git@hf.co:datasets/music-arena/music-arena-dataset
cd music-arena-dataset

# Set up Git LFS for audio files (one-time)
git lfs install
git lfs track "audio_files/**/*.mp3"

# Copy the new month's data into the dataset repo
export HF_LATEST_TAG=08-2026MAR # update this
cp -r ~/.cache/music_arena/bucket/dataset/battle_data/$HF_LATEST_TAG ./battle_data/
cp -r ~/.cache/music_arena/bucket/dataset/audio_files/$HF_LATEST_TAG ./audio_files/
cp -r ~/.cache/music_arena/bucket/dataset/metadata/$HF_LATEST_TAG.md ./metadata/

# Commit and push
git add .
git commit -m "Add $HF_LATEST_TAG data"
git push
```

#### Step 3.5: Sanity check (optional but recommended)

```bash
ma-comp leaderboard sanity-check
```

Compares local log count vs HuggingFace dataset count to verify the push was complete.

#### Step 4: Generate updated leaderboard

```bash
ma-comp leaderboard leaderboard
```

#### Step 5: Update the website

```bash
ma-comp leaderboard update-frontend
# Creates components/frontend/ma_frontend/leaderboard/{YYYYMMDD}/
# Commit and open a PR
```

### Optional: Cron job

The monthly update script can be scheduled as a cron job (not enabled by default):

```bash
crontab -e
0 0 1 * * /path/to/music-arena/components/leaderboard/monthly_update.sh >> ~/monthly_update.log 2>&1
```
