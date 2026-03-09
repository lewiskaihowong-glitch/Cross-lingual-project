"""Translate remaining English entries in safety layer datasets"""

import asyncio
import pandas as pd
import argparse
from core.translator import process_entry
from main_create_dataset import worker_pool

async def translate_safety_dataset(language, max_workers=1):
    # Load the dataset
    file_path = f"data/final/safety_layer_dataset_{language}.csv"
    df = pd.read_csv(file_path)
    
    # Find English entries that need translation
    english_mask = df['language'] == 'English'
    english_entries = df[english_mask]
    
    if len(english_entries) == 0:
        return
    
    # Create translation tasks
    tasks = []
    indices = []
    questions = []
    
    for idx, row in english_entries.iterrows():
        question = row['question']
        task = process_entry(question, language)
        tasks.append(task)
        indices.append(idx)
        questions.append(question)
    
    # Process tasks with rate limiting
    results = await worker_pool(tasks, max_workers=max_workers)
    
    # Update the dataframe with translations
    for idx, question, result in zip(indices, questions, results):
        if result and 'translated_query' in result:
            df.at[idx, 'question'] = result['translated_query']
            df.at[idx, 'language'] = 'Non-English'
    
    # Save updated dataset (replaces the original file)
    df.to_csv(file_path, index=False)

async def translate_all_datasets(max_workers=1):
    """Translate all safety layer datasets"""
    languages = ["Amharic", "Irish", "Korean", "Hindi", "Spanish"]
    
    for language in languages:
        try:
            await translate_safety_dataset(language, max_workers=max_workers)
            await asyncio.sleep(2)
        except Exception as e:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate remaining English entries in safety layer datasets"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Specific language to translate (e.g., 'Amharic'). If not specified, translates all languages.",
        default=None
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent translation workers (default: 1 to avoid rate limits)"
    )
    
    args = parser.parse_args()
    
    if args.language:
        asyncio.run(translate_safety_dataset(args.language, max_workers=args.max_workers))
    else:
        asyncio.run(translate_all_datasets(max_workers=args.max_workers))
