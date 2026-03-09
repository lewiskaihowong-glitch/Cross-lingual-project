
import argparse
import asyncio
import pandas as pd

async def get_response_from_model(question, model):
    response = await model.query(question)
    reasoning = response.get("reasoning", "No reasoning provided")
    return response, reasoning

"""Specific for kimi k2.5"""
"""We run the response generation but get the reasoning as well"""
async def reasoning_traceback(data_path, model_name):
    data = pd.read_csv(data_path)
    results = []
    for index, row in data.iterrows():
        question = row['question']
        response, reasoning = await get_response_from_model(question, model_name)
        results.append({
            "entry": index + 1,
            "query": question,
            "llm_response": response,
            "reasoning": reasoning
        })
    output_path = f"data/final/reasoning_traceback_{model_name}_{language}.json"
    pd.DataFrame(results).to_json(output_path, orient="records")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument("data_path", type=str, help="Path to the input CSV file")
    parser.add_argument("model", type=str, help="Model to test")
    parser.add_argument("language", type=str, help="Language of the dataset (e.g., 'en', 'zh')")
    args = parser.parse_args()
    language = args.language
    
    # Use the model name directly from command line
    model_name = args.model
    data_path = f"data/final/{args.data_path}"
    asyncio.run(reasoning_traceback(data_path, model_name))
    