"""Create the dataset for investigating safety layers in LLMs"""

import json

import pandas as pd

english_seed_data = pd.read_csv("data/raw/english.csv")

languages = ["Amharic", "Irish", "Korean", "Hindi", "English", "Spanish"]
languages_no_english = ["Amharic", "Irish", "Korean", "Hindi", "Spanish"]
models = ["gpt-5-mini", "kimi-k2.5", "deepseek-v3.2", "mistral-large-3", "2.5-flash"]


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

def get_failures_by_policy(model_responses):
    failures_weight = {}
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
        if failure_rate > max_failures[policy]:
            max_failures[policy] = failure_rate
        failures_weight[policy] = failure_rate
        cur_policy += 1
        start += number_per_group
    return failures_weight
    

def get_failures_per_model_language():
    failures_dict = {}
    for language in languages:
        failures_per_model = []
        failures_per_model_dict = {}
        for model in models:
            try:
                data = json.load(open(f'data/evaluation_results/evaluation_results_{model}_{language}.json', 'r', encoding='utf-8'))
                model_responses = pd.DataFrame(data["results"])            
            except Exception as e:
                model_responses = pd.read_csv(f"data/evaluation_results/evaluation_results_{model}_{language}.csv")
            
            failures_per_model_language = get_failures_by_policy(model_responses)
            failures_per_model_dict[model] = failures_per_model_language
        failures_per_model.append(failures_per_model_dict)
        failures_dict[language] = failures_per_model
    return failures_dict


def get_sampled_data_for_policy(policy, sample_size):
    data_by_policy = english_seed_data[english_seed_data['content_policy_name'] == policy]
    available_size = len(data_by_policy)
    replace = sample_size > available_size
    sample_data = data_by_policy.sample(n=sample_size, replace=replace, random_state=42)
    print(f"  Sampled {sample_size} rows from {available_size} available (replace={replace})")
    return sample_data

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
        
        print(f"Created safety layer dataset for {language} with {len(safety_layer_translated)} rows")
        safety_layer_translated.to_csv(f"data/final/safety_layer_dataset_{language}.csv", index=False)
    
def create_safety_layer_dataset(target_size=125):
    safety_layer_data = []
    rate_total = sum(max_failures.values())
    print(f"Total Failure Rate across all policies: {rate_total}")
    print(f"Target dataset size: {target_size}")
    print(max_failures)
    
    # Calculate proportions and convert to sample sizes
    for policy in max_failures.keys():
        proportion = max_failures[policy] / rate_total if rate_total > 0 else 0
        sample_size = int(proportion * target_size)
        
        print(f"Policy: {policy}, Proportion: {proportion:.3f}, Sample Size: {sample_size}")
        sampled_data = get_sampled_data_for_policy(policy, sample_size)
        safety_layer_data.append(sampled_data)
    
    final_safety_layer_dataset = pd.concat(safety_layer_data, ignore_index=True)
    print(f"\nFinal dataset size: {len(final_safety_layer_dataset)} rows")
    final_safety_layer_dataset.to_csv("data/final/safety_layer_dataset.csv", index=False)
        
    
if __name__ == "__main__":
    # failures_per_model_language = get_failures_per_model_language()
    # with open('data/failures_per_model_language.json', 'w', encoding='utf-8') as f:
    #     json.dump(failures_per_model_language, f, ensure_ascii=False, indent=4)
    # create_safety_layer_dataset()
    create_safety_layer_dataset_other_languages()
    
    