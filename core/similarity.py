import numpy as np
import pandas as pd
import json
import asyncio
from aiolimiter import AsyncLimiter
from core.query_models import EmbeddingModel, prompt_builder
from core.model_registry import get_embedding_model

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


"""
Calculates the cosine similarity between the original query and the back-translated query to assess the accuracy of the translation. 
"""
async def calculate_similarity(original_query, back_translated_query):
    embedding_model = get_embedding_model("voyage")
    
    # Get embeddings with built-in rate limiting
    original_query_embedding = await embedding_model.embed(original_query.strip())
    back_translation_embedding = await embedding_model.embed(back_translated_query.strip())
    
    similarity = cosine_similarity(original_query_embedding[0], back_translation_embedding[0])
    return similarity
