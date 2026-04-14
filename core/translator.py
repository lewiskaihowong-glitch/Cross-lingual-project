import asyncio
from core.query_models import QueryModel, EmbeddingModel
from core.similarity import calculate_similarity
from core.model_registry import get_query_model 
from prompts.translator_prompts import (
    build_back_translation_prompt,
    build_retry_translation_prompt,
    build_translation_prompt,
)

async def translation(question, language):
    translation_prompt = build_translation_prompt(question, language)
    model = get_query_model("2.0-flash")
    return await model.query(translation_prompt)

async def retry_translation(query, language):
    translation_prompt = build_retry_translation_prompt(query, language)
    model = get_query_model("2.0-flash")
    return await model.query(translation_prompt)


async def back_translation(question):
    back_translation_prompt = build_back_translation_prompt(question)
    model = get_query_model("2.0-flash")
    return await model.query(back_translation_prompt)

async def process_retry_translation(original_query, language):
        """Translation"""
        response = await retry_translation(original_query, language)

        """Back-translation"""
        back_translation_response = await back_translation(response)
        
        """Similarity calculation"""
        similarity = await calculate_similarity(original_query, back_translation_response)
        # Clamp similarity to handle floating-point precision errors
        similarity = min(max(similarity, 0.0), 1.0)
        
        if similarity >= 0.95:
            return {
                "original_query": original_query,
                "status": "Green",
                "translated_query": response,
                "similarity": similarity
            }
        else:
            return {
                "original_query": original_query,
                "status": "Red", 
                "translated_query": response,
                "similarity": similarity
            }

async def process_entry(original_query, language):
        """Translation"""
        response = await translation(original_query, language)
        print(f"Original: {original_query}\nTranslated: {response}\n")

        """Back-translation"""
        back_translation_response = await back_translation(response)
        print(f"Back-translated: {back_translation_response}\n")
        """Similarity calculation"""
        similarity = await calculate_similarity(original_query, back_translation_response)
        # Clamp similarity to handle floating-point precision errors
        similarity = min(max(similarity, 0.0), 1.0)
        if similarity >= 0.95:
            return {
                "original_query": original_query,
                "status": "Green",
                "translated_query": response,
                "similarity": similarity
            }
        elif similarity >= 0.88:
            return await process_retry_translation(original_query, language)
        else:
            return {
                "original_query": original_query,
                "status": "Red",
                "translated_query": response,
                "similarity": similarity
            }
