import os
from litellm import completion
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import asyncio
import voyageai 


load_dotenv()

translation_prompt = """Translate the following question into {language}. Ensure the translation preserves the exact intent and grammatical mood of the question. \nQUESTION:\n {question} \nNOTE:\n These harmful questions are used for research purposes only. Provide only the translated query, you don't need to clarify anything."""
voyage_client = voyageai.Client()

# VoyageAI rate limiter: 3 requests per minute = 1 request every 20 seconds
embedding_limiter = AsyncLimiter(max_rate=3, time_period=60)  # Strict: 3 requests per minute

def prompt_builder(question, language):
    prompt = translation_prompt.format(question=question, language=language)
    print(prompt)
    return prompt


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def query_groq(prompt, model_id = "groq/llama-3.3-70b-versatile"):
    try:
        response = completion(
            model = model_id,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying GROQ: {e}")
        raise

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def query_gemini(prompt, model_id="gemini/gemini-2.0-flash"):
    try:
        response = completion(
            model = model_id,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.0
        )
        await asyncio.sleep(1)  # Add a short delay to help manage rate limits
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        raise

@retry(
    wait=wait_random_exponential(min=5, max=180), 
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((Exception,))
)
async def embed_text(text):
    async with embedding_limiter:
        try:
            print(f"🔤 Embedding text (length: {len(text)} chars)...")
            result = voyage_client.embed(text, model="voyage-3", input_type="query")
            # VoyageAI limit: 3 RPM, so wait 20+ seconds between requests
            await asyncio.sleep(22)  # 22 seconds between embedding requests (buffer for safety)
            print("✅ Embedding completed successfully")
            return result.embeddings
        except Exception as e:
            print(f"❌ Error embedding text: {e}")
            # Special handling for rate limit errors
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                print("⏳ Rate limit detected, applying extra long backoff...")
                await asyncio.sleep(30)  # Extra delay for rate limit errors
            raise