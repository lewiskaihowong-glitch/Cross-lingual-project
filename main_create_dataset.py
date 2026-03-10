import asyncio
from core.translator import process_entry
import pandas as pd
import argparse
import os

async def worker_pool(tasks, max_workers=1):
    """Process tasks with controlled concurrency to avoid rate limits"""
    semaphore = asyncio.Semaphore(max_workers)
    async def worker(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[worker(task) for task in tasks])

def generateSamplesForEnglish(data, num_samples):
    group = data.groupby('content_policy_id')
    results = []
    for _, group_data in group:
        sampled = group_data.sample(n=num_samples, replace=True)
        for index, row in sampled.iterrows():
            results.append({
                "question": row['question'],
            })
    return results

def createTasksForTranslation(data, language, num_samples):
    group = data.groupby('content_policy_id') 
    tasks = []
    for _, group_data in group:
        sampled = group_data.sample(n=num_samples, replace=True)
        for index, row in sampled.iterrows():
            task = asyncio.create_task(process_entry(row['question'], language))
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

    

