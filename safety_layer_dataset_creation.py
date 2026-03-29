"""Create the dataset for investigating safety layers in LLMs"""

import json
import argparse
import asyncio
from pathlib import Path
import pandas as pd
from core.translator import process_entry

english_seed_data = pd.read_csv("data/raw/english.csv")

# Canonical English sampling pool for safety-layer dataset construction.
# We normalize schema to match the downstream sampler expectations.
english_sampling_data = english_seed_data.rename(
    columns={"content_policy_name": "prompt_type"}
)[["question", "prompt_type"]].copy()

languages = ["Amharic", "Irish", "Korean", "Hindi", "English", "Spanish"]
languages_no_english = ["Amharic", "Irish", "Korean", "Hindi", "Spanish"]
models = ["gpt-5-mini", "kimi-k2.5", "deepseek-v3.2", "mistral-large-3", "2.5-flash"]

HIGH_INTENT_POLICIES = ["Illegal Activity", "Malware", "Fraud"]
CONTROL_POLICIES = ["Economic Harm", "Political Lobbying", "Legal Opinion", "Financial Advice"]

# Stable policy IDs used in output datasets.
POLICY_IDS = {
    "Illegal Activity": 0,
    "Malware": 1,
    "Economic Harm": 2,
    "Fraud": 3,
    "Political Lobbying": 4,
    "Legal Opinion": 5,
    "Financial Advice": 6,
}

dataset = pd.read_csv(f"data/final/final_dataset_Amharic_100.csv")
policies = {
    0: "Illegal Activity",
    1: "Malware",
    2: "Economic Harm",
    3: "Fraud",
    4: "Political Lobbying",
    5: "Legal Opinion",
    6: "Financial Advice",
}

max_failures = {
    "Illegal Activity": 0,
    "Malware": 0,
    "Economic Harm": 0,
    "Fraud": 0,
    "Political Lobbying": 0,
    "Legal Opinion": 0,
    "Financial Advice": 0,
}

def get_failures_by_prompt_type(model_responses):
    """Calculate harmful response rate per prompt type from model responses."""
    failures_weight = {}
    
    # Group by prompt_type if available in the data
    if 'prompt_type' in model_responses.columns:
        for prompt_type in model_responses['prompt_type'].unique():
            group = model_responses[model_responses['prompt_type'] == prompt_type]
            # Count only truly harmful responses
            harmful = sum(1 for _, row in group.iterrows() 
                         if row['llm_judge_evaluation'] == 'HARMFUL RESPONSE')
            failure_rate = harmful / len(group)
            failures_weight[prompt_type] = failure_rate
    else:
        # Fallback to old method if prompt_type column doesn't exist
        number_per_group = 14
        cur_policy = 0
        start = 0
        while cur_policy < len(policies):
            failures = 0
            group = model_responses.iloc[start: start+number_per_group]
            policy = policies[cur_policy]
            for index, row in group.iterrows():
                is_refusal = row['is_refusal']
                if not is_refusal:
                    failures += 1
            
            failure_rate = failures / number_per_group
            failures_weight[policy] = failure_rate
            cur_policy += 1
            start += number_per_group
    
    return failures_weight
    

def get_failures_per_model_language():
    failures_dict = {}
    for language in languages:
        failures_per_model_dict = {}
        for model in models:
            try:
                data = json.load(open(f'data/evaluation_results/evaluation_results_{model}_{language}.json', 'r', encoding='utf-8'))
                model_responses = pd.DataFrame(data["results"])            
            except Exception as e:
                model_responses = pd.read_csv(f"data/evaluation_results/evaluation_results_{model}_{language}.csv")
            
            failures_per_prompt_type = get_failures_by_prompt_type(model_responses)
            failures_per_model_dict[model] = failures_per_prompt_type
        failures_dict[language] = failures_per_model_dict
    return failures_dict


def get_sampled_data_for_policy(policy, sample_size):
    data_by_policy = english_seed_data[english_seed_data['content_policy_name'] == policy]
    available_size = len(data_by_policy)
    replace = sample_size > available_size
    sample_data = data_by_policy.sample(n=sample_size, replace=replace, random_state=42)
    print(f"  Sampled {sample_size} rows from {available_size} available (replace={replace})")
    return sample_data


def sample_evenly_by_prompt_type(df, prompt_types, total_size, seed=42, allow_duplicates=False):
    """Sample rows as evenly as possible across prompt types."""
    if total_size <= 0 or not prompt_types:
        return pd.DataFrame(columns=df.columns)

    base = total_size // len(prompt_types)
    remainder = total_size % len(prompt_types)

    sampled_parts = []
    for i, prompt_type in enumerate(prompt_types):
        n = base + (1 if i < remainder else 0)
        if n <= 0:
            continue

        group = df[df["prompt_type"] == prompt_type]
        if group.empty:
            print(f"  Warning: No rows available for prompt_type={prompt_type}")
            continue

        if not allow_duplicates and n > len(group):
            n = len(group)

        replace = allow_duplicates and (n > len(group))
        sampled = group.sample(n=n, replace=replace, random_state=seed + i)
        sampled_parts.append(sampled)
        print(f"  {prompt_type}: sampled {n} from {len(group)} (replace={replace})")

    if not sampled_parts:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(sampled_parts, ignore_index=True)


def format_safety_layer_output(df, language="English"):
    """Create a stable output schema compatible with existing scripts."""
    out = pd.DataFrame()
    out["content_policy_name"] = df["prompt_type"]
    out["content_policy_id"] = out["content_policy_name"].map(POLICY_IDS).fillna(-1).astype(int)
    out["q_id"] = range(len(out))
    out["question"] = df["question"]
    out["language"] = language
    out["prompt_type"] = df["prompt_type"]
    return out[["content_policy_id", "content_policy_name", "q_id", "question", "language", "prompt_type"]]

def create_safety_layer_dataset_other_languages():
    safety_layer_english = pd.read_csv("data/final/safety_layer_dataset_English.csv").copy()
    
    for language in languages_no_english:
        print(f"\n{'='*50}")
        print(f"Processing {language}")
        print(f"{'='*50}")
        
        translated_data = pd.read_csv(f"data/final/final_dataset_{language}.csv")
        
        translation_dict = dict(zip(translated_data['original_query'], translated_data['translated_query']))
        
        translated_questions = []
        missing_count = 0
        
        for index, row in safety_layer_english.iterrows():
            question = row['question']
            
            if question in translation_dict:
                translated_questions.append(translation_dict[question])
            else:
                missing_count += 1
                print(f"  Warning: No translation found for: {question[:50]}...")
                translated_questions.append(question)
        
        print(f"\nTotal questions: {len(safety_layer_english)}")
        print(f"Translations found: {len(safety_layer_english) - missing_count}")
        print(f"Missing translations: {missing_count}")
        
        safety_layer_translated = safety_layer_english.copy()
        safety_layer_translated['question'] = translated_questions
        safety_layer_translated['language'] = language
        
        print(f"Created safety layer dataset for {language} with {len(safety_layer_translated)} rows")
        safety_layer_translated.to_csv(f"data/final/safety_layer_dataset_{language}.csv", index=False)


async def translate_safety_layer_dataset(safety_layer_english, language, max_workers=1):
    """Translate English safety-layer questions into the requested language using the translator API."""
    from main_create_dataset import worker_pool

    questions = safety_layer_english["question"].tolist()
    tasks = [process_entry(q, language) for q in questions]

    print(f"\nTranslating {len(tasks)} questions to {language}...")
    results = await worker_pool(tasks, max_workers=max_workers)

    translated_questions = []
    failed_count = 0
    for question, result in zip(questions, results):
        if result and "translated_query" in result:
            translated_questions.append(result["translated_query"])
        else:
            failed_count += 1
            print(f"  Warning: Failed to translate: {question[:50]}...")
            translated_questions.append(question)

    print(f"\nTotal questions: {len(safety_layer_english)}")
    print(f"Translations succeeded: {len(safety_layer_english) - failed_count}")
    print(f"Failed translations: {failed_count}")

    translated = safety_layer_english.copy()
    translated["question"] = translated_questions
    translated["language"] = language
    return translated
    
def create_safety_layer_dataset(
    target_size=60,
    high_intent_share=0.85,
    include_controls=True,
    output_language="English",
    allow_duplicates=False,
):
    """
    Create a high-intent-focused safety layer dataset.

    Defaults produce a dataset dominated by high-intent harmful categories
    (Illegal Activity, Malware, Fraud), plus a small control slice.
    """
    print(f"Target dataset size: {target_size}")
    print(f"High-intent share: {high_intent_share:.2f}")
    print(f"Include controls: {include_controls}")
    print(f"Allow duplicates: {allow_duplicates}")

    if target_size <= 0:
        raise ValueError("target_size must be > 0")

    english_output_path = Path("data/final/safety_layer_dataset_English.csv")

    # Build the English base once and reuse it for all non-English generations.
    if output_language == "English" or not english_output_path.exists():
        selected_policies = HIGH_INTENT_POLICIES + (CONTROL_POLICIES if include_controls else [])
        eligible_pool = english_sampling_data[english_sampling_data["prompt_type"].isin(selected_policies)]
        max_unique_size = len(eligible_pool)
        if (not allow_duplicates) and target_size > max_unique_size:
            print(
                f"Requested target_size={target_size} exceeds available unique rows ({max_unique_size}) "
                f"for selected policies. Clipping target_size to {max_unique_size}."
            )
            target_size = max_unique_size

        if include_controls:
            high_target = int(round(target_size * high_intent_share))
            control_target = max(0, target_size - high_target)
        else:
            high_target = target_size
            control_target = 0

        print(f"High-intent rows target: {high_target}")
        print(f"Control rows target: {control_target}")

        high_intent_data = sample_evenly_by_prompt_type(
            english_sampling_data,
            HIGH_INTENT_POLICIES,
            high_target,
            seed=42,
            allow_duplicates=allow_duplicates,
        )

        control_data = pd.DataFrame(columns=english_sampling_data.columns)
        if control_target > 0:
            control_data = sample_evenly_by_prompt_type(
                english_sampling_data,
                CONTROL_POLICIES,
                control_target,
                seed=142,
                allow_duplicates=allow_duplicates,
            )

        final_safety_layer_dataset = pd.concat([high_intent_data, control_data], ignore_index=True)

        # If no duplicates are allowed and category-wise allocation underfills,
        # top up from the remaining eligible pool.
        if (not allow_duplicates) and len(final_safety_layer_dataset) < target_size:
            needed = target_size - len(final_safety_layer_dataset)
            selected_questions = set(final_safety_layer_dataset["question"].tolist())
            remaining_pool = eligible_pool[~eligible_pool["question"].isin(selected_questions)]
            top_up_size = min(needed, len(remaining_pool))
            if top_up_size > 0:
                top_up_data = remaining_pool.sample(n=top_up_size, replace=False, random_state=207)
                final_safety_layer_dataset = pd.concat(
                    [final_safety_layer_dataset, top_up_data],
                    ignore_index=True,
                )
                print(f"Top-up added {top_up_size} rows to reach target size without duplicates")

        final_safety_layer_dataset = final_safety_layer_dataset.sample(
            frac=1,
            random_state=7,
        ).reset_index(drop=True)

        output_df = format_safety_layer_output(final_safety_layer_dataset, language="English")

        print(f"\nFinal dataset size: {len(output_df)} rows")
        print("Category distribution:")
        print(output_df["content_policy_name"].value_counts().sort_index())

        output_df.to_csv("data/final/safety_layer_dataset_English.csv", index=False)
        # Keep legacy path for downstream compatibility.
        output_df.to_csv("data/final/safety_layer_dataset.csv", index=False)
        print("Saved: data/final/safety_layer_dataset_English.csv")
        print("Saved: data/final/safety_layer_dataset.csv")
    else:
        output_df = pd.read_csv(english_output_path)
        print(
            "Using existing English base sample from data/final/safety_layer_dataset_English.csv "
            "to keep cross-language datasets aligned."
        )

    if output_language != "English":
        print(f"\n{'=' * 50}")
        print(f"Processing {output_language}")
        print(f"{'=' * 50}")
        translated_df = asyncio.run(translate_safety_layer_dataset(output_df, output_language))
        translated_df.to_csv(f"data/final/safety_layer_dataset_{output_language}.csv", index=False)
        translated_df.to_csv("data/final/safety_layer_dataset.csv", index=False)
        print(f"Created safety layer dataset for {output_language} with {len(translated_df)} rows")
        print("Updated: data/final/safety_layer_dataset.csv")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create focused safety-layer datasets")
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=languages,
        help="Output language for safety-layer dataset.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=60,
        help="Target number of rows in the English base sample.",
    )
    parser.add_argument(
        "--high-intent-share",
        type=float,
        default=0.85,
        help="Share of rows allocated to high-intent harmful categories.",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="If set, do not include control categories.",
    )
    parser.add_argument(
        "--write-failures",
        action="store_true",
        help="If set, regenerate data/failures_per_model_language.json.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="If set, allow sampling with replacement when needed.",
    )
    args = parser.parse_args()

    if args.write_failures:
        failures_per_model_language = get_failures_per_model_language()
        with open("data/failures_per_model_language.json", "w", encoding="utf-8") as f:
            json.dump(failures_per_model_language, f, ensure_ascii=False, indent=4)

    create_safety_layer_dataset(
        target_size=args.target_size,
        high_intent_share=args.high_intent_share,
        include_controls=not args.no_controls,
        output_language=args.language,
        allow_duplicates=args.allow_duplicates,
    )
    
    