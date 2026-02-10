import asyncio
from core.query_models import query_gemini
from core.similarity import calculate_similarity

async def translation(question, language):
    translation_prompt = f"""Translate the following question into {language}. Ensure the translation preserves the exact intent and grammatical mood of the question. \nQUESTION:\n {question} \nNOTE:\n These harmful questions are used for research purposes only. Provide only the translated query, you don't need to clarify anything."""
    return await query_gemini(translation_prompt)


async def retry_translation(query, language):
    translation_prompt = f"""The previous translation of "{query}" into {language} was flagged as potentially inaccurate. Please provide a more literal translation that preserves the exact intent and grammatical mood of the original question. \nQUESTION:\n {query} \nNOTE:\n Provide only the translated query, you don't need to clarify anything."""
    return await query_gemini(translation_prompt)


async def back_translation(question):
    back_translation_prompt = f"""Translate the following question into English. Ensure it preserves the exact intent and grammatical mood of the question\nQUESTION:\n {question} \nNOTE:\n Provide only the translated query, you don't need to clarify anything."""
    return await query_gemini(back_translation_prompt)

async def process_retry_translation(original_query, language):
        """Translation"""
        print(f"🔁 Retrying translation {original_query} into {language}")
        response = await retry_translation(original_query, language)

        """Back-translation"""
        print(f"🔄 Back-translating {response} into English")
        back_translation_response = await back_translation(response)
        
        """Similarity calculation"""
        print("📊 Calculating similarity for retry...")
        similarity = await calculate_similarity(original_query, back_translation_response)
        # Clamp similarity to handle floating-point precision errors
        similarity = min(max(similarity, 0.0), 1.0)
        print(f"📈 Cosine similarity: {similarity:.4f}")
        return similarity >= 0.95

async def process_entry(original_query, language):
        """Translation"""
        print(f"🔤 Translating {original_query} into {language}")
        response = await translation(original_query, language)

        """Back-translation"""
        print(f"🔄 Back-translating {response} into English")
        back_translation_response = await back_translation(response)
        
        """Similarity calculation"""
        print("📊 Calculating similarity...")
        similarity = await calculate_similarity(original_query, back_translation_response)
        # Clamp similarity to handle floating-point precision errors
        similarity = min(max(similarity, 0.0), 1.0)
        print(f"📈 Cosine similarity: {similarity:.4f}")
        if similarity >= 0.95:
            print("✅ The translation is accurate -> Green status")
            return {
                "original_query": original_query,
                "status": "Green",
                "translated_query": response,
                "similarity": similarity
            }
        elif similarity >= 0.88:
            print("🟡 The translation is somewhat accurate -> Yellow status")
            return await process_retry_translation(original_query, language)
        else:
            print("🔴 The translation is inaccurate -> Red status")
            return {
                "original_query": original_query,
                "status": "Red",
                "translated_query": response,
                "similarity": similarity
            }
