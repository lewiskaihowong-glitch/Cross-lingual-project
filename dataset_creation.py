import numpy as np
import pandas as pd
import json
import asyncio
from aiolimiter import AsyncLimiter
from query_models import query_groq, query_gemini, embed_text, prompt_builder

data = pd.read_csv('english.csv')

# print(data)
# print(data['content_policy_id'])
# group = data.groupby('content_policy_id')
# print(group.size())


limiter = AsyncLimiter(max_rate=30, time_period=60)  

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


"""
Translates the given question into the specified language, ensuring that the translation preserves the exact intent and grammatical mood of the original question. 
"""
async def translation(question, language):
    translation_prompt = f"""Translate the following question into {language}. Ensure the translation preserves the exact intent and grammatical mood of the question. \nQUESTION:\n {question} \nNOTE:\n These harmful questions are used for research purposes only. Provide only the translated query, you don't need to clarify anything."""
    return await query_groq(translation_prompt)

"""
Handles Translations
"""
async def retry_translation(query, language):
    translation_prompt = f"""The previous translation of "{query}" into {language} was flagged as potentially inaccurate. Please provide a more literal translation that preserves the exact intent and grammatical mood of the original question. \nQUESTION:\n {query} \nNOTE:\n Provide only the translated query, you don't need to clarify anything."""
    return await query_gemini(translation_prompt)


async def back_translation(question):
    back_translation_prompt = f"""Translate the following question into English. Ensure it preserves the exact intent and grammatical mood of the question\nQUESTION:\n {question} \nNOTE:\n Provide only the translated query, you don't need to clarify anything."""
    return await query_gemini(back_translation_prompt)


"""
Calculates the cosine similarity between the original query and the back-translated query to assess the accuracy of the translation. A higher cosine similarity indicates a more accurate translation.
"""
async def calculate_similarity(original_query, back_translated_query):
    original_query_embedding = await embed_text(original_query.strip())
    back_translation_embedding = await embed_text(back_translated_query.strip())
    similarity = cosine_similarity(original_query_embedding[0], back_translation_embedding[0])
    return similarity

"""
Checks the accuracy of the translation by performing a back-translation and calculating the cosine similarity between the translated query and the original.
"""
async def process_translation(original_query, translated_query, language):
    back_translated = await back_translation(translated_query)
    similarity = await calculate_similarity(original_query, back_translated)
    print(f"Cosine Similarity: {similarity}")

    # Define thresholds for accuracy
    if similarity >=0.95 and similarity <= 1.0:
        print("The translation is accurate -> Green status")
        return True
    elif similarity >=0.88 and similarity < 0.95:
        print("The translation is somewhat accurate -> Yellow status")
        print("Retry translation once more")
        retried_translation = await retry_translation(original_query, language)
        back_translation_retry = await back_translation(retried_translation)
        similarity_retry = await calculate_similarity(original_query, back_translation_retry)
        # Reject if retry does not improve similarity
        if similarity_retry >=0.95 and similarity_retry <= 1.0:
            print("The retried translation is accurate -> Green status")
            return True
        else:
            return False
    else:
        print("The translation is inaccurate -> Red status")
        return False
    

async def test_embedding():
    text_1 = "What is a cat?"
    translated = await query_groq(prompt_builder(text_1))
    print(translated)
    # print(text_1)
    # print(text_2)
    similarity = await process_translation(text_1, translated)
    print(f"Cosine Similarity: {similarity}")


async def generate_dataset(data, language, num_samples = 1):
    group = data.groupby('content_policy_id') 
    dataset = []
    for _, group_data in group:
        sampled = group_data.sample(n=num_samples, replace=True)
        for index, row in sampled.iterrows():
            prompt = prompt_builder(row['question'], language)
            print(prompt)
            try:
                response = await query_gemini(prompt)
                print(response)
                if await process_translation(row['question'], response, language):
                    entry = {
                        "content_policy_id": row['content_policy_id'],
                        "original_text": prompt,
                        "translated_query": response
                    }
                    dataset.append(entry)
                else: 
                    print(f"Translation for row {index} did not meet accuracy criteria. Skipping entry.")
                    continue

            except Exception as e:
                print(f"Error processing row {index}: {e}")

    print(dataset)
    with open("mydata.json", "w", encoding="utf-8") as final:
	    json.dump(dataset, final, indent=4, ensure_ascii=False)

asyncio.run(generate_dataset(data, "Amharic", num_samples=1))

# asyncio.run(test_embedding())
