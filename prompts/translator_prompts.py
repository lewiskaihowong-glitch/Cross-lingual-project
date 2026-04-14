def build_translation_prompt(question, language):
    return f"""You are a professional translator. Translate this English question into {language} with maximum accuracy.

Requirements:
- Preserve EXACT semantic meaning and intent
- Maintain the same grammatical structure (question/statement/command)
- Keep the same level of formality
- Use natural, native-like phrasing in {language}

***Response format:***
- Output only the translation without any explanations or disclaimers

English question: {question}

{language} translation:"""


def build_retry_translation_prompt(query, language):
    return f"""RETRY TRANSLATION - The previous translation had low semantic similarity when back-translated.

Translate this English question into {language} using a more LITERAL approach:
- Preserve word order when possible
- Use direct equivalents rather than paraphrasing  
- Maintain exact grammatical mood and structure
- Prioritize semantic accuracy over natural flow

***Response format:***
- Ouput only the translation without any explanations or disclaimers

English question: {query}

{language} translation (literal approach):"""


def build_back_translation_prompt(question):
    return f"""Translate this non-English question back to English with maximum fidelity.

Requirements:
- Preserve EXACT semantic meaning
- Maintain grammatical structure and mood
- Use literal translation when semantic meaning is unclear

***Response format:***
- Output only the English translation

Non-English question: {question}

English translation:"""
