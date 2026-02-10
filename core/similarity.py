import numpy as np
import pandas as pd
import json
import asyncio
from aiolimiter import AsyncLimiter
from core.query_models import query_groq, query_gemini, embed_text, prompt_builder

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


"""
Calculates the cosine similarity between the original query and the back-translated query to assess the accuracy of the translation. 
"""
async def calculate_similarity(original_query, back_translated_query):
    # Get embeddings with built-in rate limiting
    original_query_embedding = await embed_text(original_query.strip())
    back_translation_embedding = await embed_text(back_translated_query.strip())
    
    similarity = cosine_similarity(original_query_embedding[0], back_translation_embedding[0])
    return similarity
