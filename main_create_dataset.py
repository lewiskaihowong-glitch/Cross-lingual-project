import asyncio
from core.translator import process_entry
import pandas as pd
import argparse
import os


def stratified_unique_sample(data, num_samples, group_col="content_policy_id"):
    """Sample up to num_samples per group without replacement and without duplicate source rows."""
    if group_col not in data.columns:
        raise KeyError(f"Missing required grouping column: {group_col}")

    # q_id can repeat across groups, so include group key when available.
    if "q_id" in data.columns:
        dedupe_cols = [group_col, "q_id"]
    else:
        dedupe_cols = [group_col, "question"]
    unique_data = data.drop_duplicates(subset=dedupe_cols)

    sampled_groups = []
    for group_value, group_data in unique_data.groupby(group_col):
        sample_size = min(num_samples, len(group_data))
        if sample_size < num_samples:
            print(
                f"Group {group_value}: requested {num_samples} but only {len(group_data)} unique rows available; taking {sample_size}."
            )
        sampled_groups.append(group_data.sample(n=sample_size, replace=False, random_state=42))

    if not sampled_groups:
        return unique_data.iloc[0:0].copy()

    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    sampled_df = sampled_df.drop_duplicates(subset=dedupe_cols)
    return sampled_df

async def worker_pool(tasks, max_workers=1):
    """Process tasks with controlled concurrency to avoid rate limits"""
    semaphore = asyncio.Semaphore(max_workers)
    async def worker(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[worker(task) for task in tasks])

def generateSamplesForEnglish(data, num_samples):
    sampled_data = stratified_unique_sample(data, num_samples)
    results = []
    for _, row in sampled_data.iterrows():
        results.append(
            {
                "question": row["question"],
                "prompt_type": row.get("content_policy_name", "UNKNOWN"),
            }
        )
    return results

async def process_entry_with_metadata(row, language):
    result = await process_entry(row["question"], language)
    if result is None:
        return None
    result["prompt_type"] = row.get("content_policy_name", "UNKNOWN")
    return result


def createTasksForTranslationNoGrouping(data, language):
    tasks = []
    for _, row in data.iterrows():
        task = asyncio.create_task(process_entry_with_metadata(row, language))
        tasks.append(task)
    return tasks

def createTasksForTranslation(data, language, num_samples):
    sampled_data = stratified_unique_sample(data, num_samples)
    tasks = []
    for _, row in sampled_data.iterrows():
        task = asyncio.create_task(process_entry_with_metadata(row, language))
        tasks.append(task)
    return tasks

async def run_pipeline(data_path, language, num_samples):
    print(f"Starting pipeline with: data_path={data_path}, language={language}, num_samples={num_samples}")
    data = pd.read_csv(data_path)
    print(f"Loaded data with {len(data)} rows")
    
    if language == "English":
        results = generateSamplesForEnglish(data, num_samples)
        print(f"Generated {len(results)} samples for English")
    else:
        tasks = createTasksForTranslation(data, language, num_samples)
        print(f"Created {len(tasks)} tasks")
        results = await worker_pool(tasks, max_workers=1)
        print(f"Completed {len(results)} tasks")
    
    # Filter valid results
    valid_results = [result for result in results if result is not None]
    print(f"Valid results: {len(valid_results)}")
    
    if valid_results:
        final_dataset = pd.DataFrame(valid_results)
        os.makedirs("data/final", exist_ok=True)
        output_path = f"data/final/final_dataset_{language}.csv"
        final_dataset.to_csv(output_path, index=False)
        print(f"Saved {len(final_dataset)} entries to {output_path}")
        
        # Verify file was created
        if os.path.exists(output_path):
            print(f"File exists: {output_path} ({os.path.getsize(output_path)} bytes)")
        else:
            print(f"ERROR: File was not created at {output_path}")
    else:
        print("ERROR: No valid results to save!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the translation and similarity pipeline.")
    parser.add_argument("data_path", type=str, help="Path to the input CSV file")
    parser.add_argument("language", type=str, help="Target language for translation")
    parser.add_argument("num_samples", type=int, help="Number of samples to take from ")
    args = parser.parse_args()
                        
    data_path = f"data/final/{args.data_path}"
    language = args.language
    num_samples = args.num_samples
    asyncio.run(run_pipeline(data_path, language, num_samples))

    

