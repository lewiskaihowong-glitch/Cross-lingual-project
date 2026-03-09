"""Clean safety layer datasets by removing explanation text"""

import pandas as pd
import re

languages = ["Amharic", "Irish", "Korean", "Hindi", "English", "Spanish"]

def clean_question(question):
    """Remove explanation text from questions"""
    if pd.isna(question):
        return question
    
    # Convert to string
    question_str = str(question)
    
    # Patterns to match explanation sections
    patterns = [
        r'\*\*Explanation.*',  # Matches **Explanation...
        r'\nExplanation.*',    # Matches newline Explanation...
        r'\n\*.*',             # Matches newline followed by bullet points
    ]
    
    # Remove explanation sections
    for pattern in patterns:
        question_str = re.split(pattern, question_str, flags=re.IGNORECASE | re.DOTALL)[0]
    
    # Clean up extra whitespace and newlines
    question_str = question_str.strip()
    
    return question_str

def clean_safety_datasets():
    for language in languages:
        file_path = f"data/final/safety_layer_dataset_{language}.csv"
        try:
            print(f"\nProcessing {language}...")
            df = pd.read_csv(file_path)
            
            # Show example before cleaning
            print(f"  Before: {len(df)} rows")
            sample_idx = df['question'].astype(str).str.contains('Explanation', case=False, na=False)
            print(f"  Questions with explanations: {sample_idx.sum()}")
            
            # Clean questions
            df['question'] = df['question'].apply(clean_question)
            
            # Save cleaned dataset
            df.to_csv(file_path, index=False)
            print(f"  Cleaned and saved {language}")
            
        except FileNotFoundError:
            print(f"  File not found: {file_path}")
        except Exception as e:
            print(f"  Error processing {language}: {e}")

if __name__ == "__main__":
    clean_safety_datasets()
    print("\n✓ All datasets cleaned!")
