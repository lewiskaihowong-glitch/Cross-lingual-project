from query_models import Model

prompt = "What is a cat"
model_llama = Model("groq/llama-3.3-70b-versatile")
model_gemini = Model("gemini/gemini-2.0-flash")

print(model_gemini.query(prompt))

