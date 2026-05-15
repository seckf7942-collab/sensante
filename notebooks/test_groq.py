# notebooks/test_groq.py
# Exercice 2 : Tester differentes temperatures

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM = """Tu es un assistant medical senegalais.
Explique le diagnostic en francais simple.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-meme."""

USER = """Patient : Homme, 35 ans, region Thies
Temperature : 38.8 C
Diagnostic du modele : grippe (probabilite 65%)
Explique ce resultat au patient."""

for temp in [0.0, 0.5, 1.0]:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER}
        ],
        max_tokens=200,
        temperature=temp
    )
    print(f"\n=== Temperature = {temp} ===")
    print(response.choices[0].message.content)