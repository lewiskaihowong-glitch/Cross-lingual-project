import asyncio
from core.query_models import QueryModel, EmbeddingModel
from core.similarity import calculate_similarity
from core.model_registry import get_query_model 

async def translation(question, language):
    translation_prompt = f"""You are a professional translator. Translate this English question into {language} with maximum accuracy.

Requirements:
- Preserve EXACT semantic meaning and intent
- Maintain the same grammatical structure (question/statement/command)
- Keep the same level of formality
- Use natural, native-like phrasing in {language}
- Do not add explanations, disclaimers, or extra content, provide only the translation

English question: {question}

{language} translation:"""
    model = get_query_model("2.0-flash")
    return await model.query(translation_prompt)

async def retry_translation(query, language):
    translation_prompt = f"""RETRY TRANSLATION - The previous translation had low semantic similarity when back-translated.

Translate this English question into {language} using a more LITERAL approach:
- Preserve word order when possible
- Use direct equivalents rather than paraphrasing  
- Maintain exact grammatical mood and structure
- Prioritize semantic accuracy over natural flow
- Do not add explanations, disclaimers, or extra content, provide only the translation

English question: {query}

{language} translation (literal approach):"""
    model = get_query_model("2.0-flash")
    return await model.query(translation_prompt)


async def back_translation(question):
    back_translation_prompt = f"""Translate this non-English question back to English with maximum fidelity.

Requirements:
- Preserve EXACT semantic meaning
- Maintain grammatical structure and mood
- Use literal translation when semantic meaning is unclear
- Output only the English translation

Non-English question: {question}

English translation:"""
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
