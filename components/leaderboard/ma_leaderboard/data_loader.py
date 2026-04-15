"""Download and parse battle logs from GCP.

Authenticates via the GCP_BUCKET_SERVICE_ACCOUNT secret (same service
account JSON used by the gateway component). Bucket names are read
via music_arena.secret.

Install with: pip install -e "components/leaderboard/[gcp]"
"""

import json
import os

import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

from .config import MODELS_METADATA


def _get_gcp_client():
    """Create a GCP storage client using the shared service account secret."""
    from google.cloud import storage
    from music_arena.secret import get_secret, get_secret_json

    credentials = get_secret_json("GCP_BUCKET_SERVICE_ACCOUNT")
    metadata_bucket = get_secret("GCP_METADATA_BUCKET").strip()
    audio_bucket = get_secret("GCP_AUDIO_BUCKET").strip()

    client = storage.Client.from_service_account_info(credentials)
    return client, metadata_bucket, audio_bucket


def download_filtered_logs_and_audio(
    logs_dir, audio_dir, start_date=None, end_date=None, max_workers=16
):
    """Download battle logs and audio from GCP buckets.

    Filters by known models and optional date range.
    Skips files already present locally.
    Downloads logs and audio in parallel using a thread pool.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import urlparse

    client, metadata_bucket_name, audio_bucket_name = _get_gcp_client()

    print(f"Starting integrated download (workers={max_workers})...")
    print(f" - Logs will be saved to: '{logs_dir}'")
    print(f" - Audio will be saved to: '{audio_dir}'")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    metadata_bucket = client.bucket(metadata_bucket_name)
    audio_bucket = client.bucket(audio_bucket_name)

    local_log_files = set(os.listdir(logs_dir))
    local_audio_files = set(os.listdir(audio_dir))
    known_models = set(MODELS_METADATA.keys())

    all_blobs = list(metadata_bucket.list_blobs())
    if start_date and end_date:
        all_blobs = [
            b
            for b in all_blobs
            if b.time_created and start_date <= b.time_created <= end_date
        ]

    # Filter to only new JSON blobs
    new_blobs = [
        b for b in all_blobs
        if b.name.endswith(".json")
        and os.path.basename(b.name) not in local_log_files
    ]

    print(
        f"Found {len(all_blobs)} logs in range, "
        f"{len(all_blobs) - len(new_blobs)} already local, "
        f"{len(new_blobs)} to download."
    )

    # Phase 1: Download and filter log blobs in parallel
    def _download_log(blob):
        """Download a single log blob. Returns (filename, content, audio_urls) or None."""
        try:
            content = blob.download_as_string()
            data = json.loads(content)

            if not data.get("vote"):
                return None

            model_a = (
                data.get("a_metadata", {})
                .get("system_key", {})
                .get("system_tag")
            )
            model_b = (
                data.get("b_metadata", {})
                .get("system_key", {})
                .get("system_tag")
            )

            if not (
                model_a
                and model_b
                and model_a in known_models
                and model_b in known_models
            ):
                return "skipped_unknown"

            audio_urls = [data.get("a_audio_url"), data.get("b_audio_url")]
            return (os.path.basename(blob.name), content, audio_urls)
        except Exception as e:
            print(f"\nWarning: Failed to download {blob.name}: {e}")
            return None

    downloaded_logs = 0
    skipped_unknown = 0
    audio_to_download = []  # list of audio filenames to fetch

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_log, b): b for b in new_blobs}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading logs"
        ):
            result = future.result()
            if result is None:
                continue
            if result == "skipped_unknown":
                skipped_unknown += 1
                continue

            log_filename, content, audio_urls = result
            with open(os.path.join(logs_dir, log_filename), "wb") as f:
                f.write(content)
            downloaded_logs += 1

            for url in audio_urls:
                if not url:
                    continue
                audio_filename = os.path.basename(urlparse(url).path)
                if audio_filename and audio_filename not in local_audio_files:
                    audio_to_download.append(audio_filename)
                    local_audio_files.add(audio_filename)

    # Phase 2: Download audio files in parallel
    # Deduplicate (two logs could reference the same audio)
    audio_to_download = list(dict.fromkeys(audio_to_download))

    def _download_audio(audio_filename):
        """Download a single audio file. Returns True on success."""
        try:
            audio_blob = audio_bucket.blob(audio_filename)
            if audio_blob.exists():
                dest_path = os.path.join(audio_dir, audio_filename)
                audio_blob.download_to_filename(dest_path)
                return True
        except Exception as e:
            print(f"\nWarning: Failed to download audio {audio_filename}: {e}")
        return False

    downloaded_audio = 0
    if audio_to_download:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_download_audio, f): f for f in audio_to_download
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Downloading audio",
            ):
                if future.result():
                    downloaded_audio += 1

    print(f"\nDownload complete.")
    print(f"  Logs downloaded: {downloaded_logs}")
    print(f"  Audio downloaded: {downloaded_audio}")
    print(f"  Skipped (unknown models): {skipped_unknown}")


def load_all_raw_logs(log_dir: str) -> list:
    """Load all JSON log files from a directory."""
    if not os.path.exists(log_dir):
        return []
    raw_logs = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".json"):
            with open(os.path.join(log_dir, filename), "r") as f:
                try:
                    raw_logs.append(json.load(f))
                except json.JSONDecodeError:
                    continue
    return raw_logs


def parse_logs(log_dir, start_date=None, end_date=None):
    """Parse local JSON logs into a battles DataFrame.

    Returns:
        Tuple of (battles_df, raw_logs)
    """
    print(f"\nParsing logs from: {log_dir}")

    parsed_data = []
    raw_logs = []
    known_models = set(MODELS_METADATA.keys())
    skipped_unknown = 0

    log_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]

    for filename in tqdm(log_files, desc="Parsing files"):
        filepath = os.path.join(log_dir, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        # Recover prompt from detailed prompt if missing
        if not data.get("prompt") and data.get("prompt_prebaked"):
            prompt_detailed = data.get("prompt_detailed")
            if isinstance(prompt_detailed, dict):
                recovered = prompt_detailed.get("overall_prompt")
                if not recovered:
                    lyrics = data.get("a_metadata", {}).get(
                        "lyrics"
                    ) or data.get("b_metadata", {}).get("lyrics")
                    if lyrics:
                        recovered = f"(Lyrics) {lyrics[:100]}..."
                if recovered:
                    data["prompt"] = recovered

        # Date filtering
        if start_date and end_date:
            session_time_unix = None
            prompt_session = data.get("prompt_session")
            if isinstance(prompt_session, dict):
                session_time_unix = prompt_session.get("create_time")
            if not session_time_unix:
                a_meta = data.get("a_metadata")
                if isinstance(a_meta, dict):
                    session_time_unix = a_meta.get("gateway_time_completed")

            if session_time_unix:
                try:
                    ts_dt = datetime.fromtimestamp(
                        session_time_unix, tz=timezone.utc
                    )
                    if not (start_date <= ts_dt <= end_date):
                        continue
                except (ValueError, OSError):
                    continue
            else:
                continue

        # Normalize instrumental boolean
        prompt_detailed = data.get("prompt_detailed")
        if not isinstance(prompt_detailed, dict):
            prompt_detailed = {}
            data["prompt_detailed"] = prompt_detailed

        raw_inst = prompt_detailed.get("instrumental")
        is_inst = False
        if raw_inst is not None:
            if isinstance(raw_inst, str):
                is_inst = raw_inst.lower() == "true"
            else:
                is_inst = bool(raw_inst)
        data["prompt_detailed"]["instrumental"] = is_inst

        # Extract lyrics
        extracted_lyrics = data.get("a_metadata", {}).get("lyrics")
        if not extracted_lyrics:
            extracted_lyrics = data.get("b_metadata", {}).get("lyrics")
        data["lyrics"] = extracted_lyrics

        raw_logs.append(data)

        if (
            data.get("vote")
            and data.get("a_metadata")
            and data.get("b_metadata")
        ):
            try:
                model_a = data["a_metadata"]["system_key"]["system_tag"]
                model_b = data["b_metadata"]["system_key"]["system_tag"]

                if model_a not in known_models or model_b not in known_models:
                    skipped_unknown += 1
                    continue

                pref = data["vote"]["preference"]
                if pref == "BOTH_BAD":
                    continue  # Exclude BOTH_BAD from scoring
                winner = "tie"
                if pref == "A":
                    winner = "model_a"
                elif pref == "B":
                    winner = "model_b"

                parsed_data.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "winner": winner,
                        "duration_a": data["a_metadata"]["duration"],
                        "generation_time_a": data["a_metadata"][
                            "gateway_time_completed"
                        ]
                        - data["a_metadata"]["gateway_time_started"],
                        "duration_b": data["b_metadata"]["duration"],
                        "generation_time_b": data["b_metadata"][
                            "gateway_time_completed"
                        ]
                        - data["b_metadata"]["gateway_time_started"],
                        "is_instrumental": is_inst,
                        "instrumental": is_inst,
                        "lyrics": extracted_lyrics,
                    }
                )
            except (KeyError, TypeError):
                continue

    if skipped_unknown:
        print(f"  Skipped {skipped_unknown} battles with unknown models")

    return pd.DataFrame(parsed_data), raw_logs
