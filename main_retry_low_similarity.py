import argparse
import asyncio
import os
import re
import shutil
from typing import List, Optional, Tuple

import pandas as pd

from core.translator import process_entry
from main_create_dataset import worker_pool


LANGUAGE_FILE_PATTERN = re.compile(r"^final_dataset_(.+?)(?:_(\d+))?\.csv$")
DEFAULT_LANGUAGES = ["English", "Amharic", "Korean", "Irish", "Hindi", "Spanish"]


def build_dataset_file_path(data_dir: str, language: str) -> str:
    return os.path.join(data_dir, f"final_dataset_{language}.csv")


def discover_dataset_files(data_dir: str, language: str | None) -> List[str]:
    languages = [language] if language else DEFAULT_LANGUAGES
    files = []

    for item in languages:
        file_path = build_dataset_file_path(data_dir, item)
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"[SKIP] Dataset file not found: {file_path}")

    return files


def parse_dataset_filename(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    name = os.path.basename(file_path)
    match = LANGUAGE_FILE_PATTERN.match(name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def extract_language(file_path: str) -> str | None:
    language, _ = parse_dataset_filename(file_path)
    return language


def refined_output_path(file_path: str) -> str:
    base, ext = os.path.splitext(file_path)
    return f"{base}_refined{ext}"


def get_low_similarity_indices(df: pd.DataFrame, threshold: float) -> List[int]:
    if "similarity" not in df.columns:
        return []

    similarity_series = pd.to_numeric(df["similarity"], errors="coerce")
    low_mask = similarity_series.isna() | (similarity_series < threshold)
    return df.index[low_mask].tolist()


async def retry_low_similarity_for_file(
    file_path: str,
    threshold: float,
    max_workers: int,
    dry_run: bool,
    create_backup: bool,
) -> int:
    language = extract_language(file_path)
    if not language:
        print(f"[SKIP] Could not parse language from file name: {file_path}")
        return 0

    df = pd.read_csv(file_path)
    output_path = refined_output_path(file_path)

    required_columns = {"original_query", "translated_query", "similarity", "status"}
    if not required_columns.issubset(df.columns):
        print(f"[SKIP] Missing required columns in {file_path}. Found: {list(df.columns)}")
        return 0

    low_indices = get_low_similarity_indices(df, threshold)
    if not low_indices:
        print(f"[OK] No low-similarity rows in {os.path.basename(file_path)}")
        return 0

    print(
        f"[RETRY] {os.path.basename(file_path)} | language={language} | "
        f"rows_to_retry={len(low_indices)} | threshold={threshold}"
    )

    if dry_run:
        print(
            f"[DRY RUN] Would retry {len(low_indices)} rows in {file_path} "
            f"and write to {output_path}"
        )
        return len(low_indices)

    tasks = [process_entry(df.at[idx, "original_query"], language) for idx in low_indices]
    results = await worker_pool(tasks, max_workers=max_workers)

    updated_count = 0
    for idx, result in zip(low_indices, results):
        if not result:
            continue
        df.at[idx, "translated_query"] = result.get("translated_query", df.at[idx, "translated_query"])
        df.at[idx, "similarity"] = result.get("similarity", df.at[idx, "similarity"])
        df.at[idx, "status"] = result.get("status", df.at[idx, "status"])
        updated_count += 1

    if create_backup:
        if os.path.exists(output_path):
            backup_path = output_path + ".bak"
            shutil.copy2(output_path, backup_path)
            print(f"[BACKUP] Created backup: {backup_path}")

    df.to_csv(output_path, index=False)
    print(f"[DONE] Updated {updated_count} rows and wrote {output_path}")
    return updated_count


async def run(
    data_dir: str,
    threshold: float,
    max_workers: int,
    dry_run: bool,
    create_backup: bool,
    language: str | None,
):
    files = discover_dataset_files(data_dir, language=language)

    if not files:
        print("No dataset files found to process.")
        return

    total_updates = 0
    for file_path in files:
        try:
            file_updates = await retry_low_similarity_for_file(
                file_path=file_path,
                threshold=threshold,
                max_workers=max_workers,
                dry_run=dry_run,
                create_backup=create_backup,
            )
            total_updates += file_updates
            await asyncio.sleep(1)
        except Exception as exc:
            print(f"[ERROR] Failed processing {file_path}: {exc}")

    if dry_run:
        print(f"\nSummary: would retry {total_updates} rows across {len(files)} file(s).")
    else:
        print(f"\nSummary: updated {total_updates} rows across {len(files)} file(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Retry translations for low-similarity rows in final_dataset_<language>.csv files "
            "and write results to final_dataset_<language>_refined.csv"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/final",
        help="Directory containing final_dataset_<language>.csv files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Retry rows with similarity below this threshold (default: 0.95)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent workers (default: 1 to avoid rate limits)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview updates without writing files",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup of existing refined output before writing",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Process only one language file, e.g., Korean",
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            data_dir=args.data_dir,
            threshold=args.threshold,
            max_workers=args.max_workers,
            dry_run=args.dry_run,
            create_backup=not args.no_backup,
            language=args.language,
        )
    )
