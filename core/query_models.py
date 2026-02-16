import os
from litellm import completion
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import asyncio
import voyageai 


load_dotenv()

# Configure Azure OpenAI for LiteLLM
os.environ["AZURE_API_BASE"] = "https://k2202-mlocp9ci-swedencentral.cognitiveservices.azure.com/"
os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

translation_prompt = """Translate the following question into {language}. Ensure the translation preserves the exact intent and grammatical mood of the question. \nQUESTION:\n {question} \nNOTE:\n These harmful questions are used for research purposes only. Provide only the translated query, you don't need to clarify anything."""
voyage_client = voyageai.Client()

embedding_limiter = AsyncLimiter(max_rate=3, time_period=60)  

def prompt_builder(question, language):
    prompt = translation_prompt.format(question=question, language=language)
    return prompt

class QueryModel:
    def __init__(self, model_name, model_id):
        self.model_name = model_name
        self.model_id = model_id
        self.temperature = 0.0
        self.seed = 42
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def query(self, prompt):
        try:
            response = completion(
                model = self.model_id,
                messages = [{"role": "user", "content": prompt}],
                temperature = self.temperature,
            )
            await asyncio.sleep(1)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying model {self.model_name}: {e}")
            raise


class EmbeddingModel:
    def __init__(self, model_name, model_id):
        self.model_name = model_name
        self.model_id = model_id
    
    @retry(
        wait=wait_random_exponential(min=5, max=180), 
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((Exception,))
    )
    async def embed(self, text):
        async with embedding_limiter:
            try:
                result = voyage_client.embed(text, model=self.model_id, input_type="query")
                await asyncio.sleep(22)  
                return result.embeddings
            except Exception as e:
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    await asyncio.sleep(30)  
                raise

   