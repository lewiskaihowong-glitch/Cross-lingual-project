import asyncio
from core.translator import process_entry
import pandas as pd
import argparse

async def worker_pool(tasks, max_workers=1):
    """Process tasks with controlled concurrency to avoid rate limits"""
    semaphore = asyncio.Semaphore(max_workers)
    async def worker(task):
        async with semaphore:
            return await task
    
    print(f"🔄 Processing {len(tasks)} tasks with max {max_workers} concurrent workers...")
    print("⚠️  VoyageAI limits: 3 RPM, 10k TPM - This will take a while!")
    print("🐌 ~22 seconds minimum between each embedding request...")
    return await asyncio.gather(*[worker(task) for task in tasks])

async def run_pipeline(data_path, language, num_samples):
    print(data_path)
    data = pd.read_csv(data_path)
    group = data.groupby('content_policy_id') 
    tasks = []
    for _, group_data in group:
        sampled = group_data.sample(n=num_samples, replace=True)
        for index, row in sampled.iterrows():
            task = asyncio.create_task(process_entry(row['question'], language))
            tasks.append(task)

    total_tasks = len(tasks)
    # Each task requires 2 embeddings (original + back-translation)
    # With 3 RPM limit and 22 second delays, estimate processing time
    estimated_minutes = (total_tasks * 2 * 22) / 60
    print(f"📊 Processing {total_tasks} entries...")
    print(f"⏱️  Each entry requires 2 embeddings with VoyageAI rate limits (3 RPM)")
    print(f"🕒 Estimated processing time: ~{estimated_minutes:.1f} minutes")
    print(f"☕ This will be slow due to strict rate limits - grab some coffee!")
    
    results = await worker_pool(tasks, max_workers=1)
    final_dataset = pd.DataFrame([result for result in results if result])
    final_dataset.to_csv(f"data/final/final_dataset_{language}.csv", index=False)
    print(f"✅ Final dataset for {language} created with {len(final_dataset)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the translation and similarity pipeline.")
    parser.add_argument("data_path", type=str, help="Path to the input CSV file")
    parser.add_argument("language", type=str, help="Target language for translation")
    parser.add_argument("num_samples", type=int, help="Number of samples to take from ")
    args = parser.parse_args()
                        
    data_path = f"data/raw/{args.data_path}"
    language = args.language
    num_samples = args.num_samples
    asyncio.run(run_pipeline(data_path, language, num_samples))

    

