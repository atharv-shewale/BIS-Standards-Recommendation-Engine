import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMGenerator:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not found in environment. Using MOCK mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            self.client = Groq(api_key=api_key)
            self.model_name = model_name

    def generate_final_results(self, query, candidates):
        """
        Takes candidates and uses LLM to select and explain the top 5.
        Deterministic: only uses standard_ids provided by the retriever.
        """
        if self.mock_mode:
            # Fallback mock logic
            ranked = candidates[:5]
            explanations = {c["standard_id"]: f"Factual recommendation for {query[:20]}... based on retrieved standard {c['standard_id']}." for c in ranked}
            return [c["standard_id"] for c in ranked], explanations
        
        candidate_context = "\n".join([f"[{i}] {c['standard_id']}: {c['text'][:500]}" for i, c in enumerate(candidates)])
        
        prompt = f"""
You are a BIS Standards Compliance Intelligence Engine. 
Given the user query and the list of retrieved standards below, select the TOP 5 most relevant standards.
For each selected standard, provide a concise 1-2 line justification.

User Query: {query}

Retrieved Standards:
{candidate_context}

CRITICAL: 
- Respond ONLY with a JSON object.
- Strictly select from the retrieved standards provided.
- Format: {{"selected_standards": ["IS 1234", "IS 5678", ...], "justifications": {{"IS 1234": "...", "IS 5678": "..."}}}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a BIS Standards Compliance Intelligence Engine. Respond ONLY with JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            clean_text = response.choices[0].message.content.strip()
            data = json.loads(clean_text)
            
            selected = data.get("selected_standards", [])[:5]
            justifications = data.get("justifications", {})
            
            # Ensure we have justifications for all selected
            for std in selected:
                if std not in justifications:
                    justifications[std] = "Factual recommendation based on compliance requirements."
                    
            return selected, justifications
        except Exception as e:
            print(f"Groq LLM Ranking Error: {e}")
            ranked = candidates[:5]
            explanations = {c["standard_id"]: "Factual recommendation based on retrieved context." for c in ranked}
            return [c["standard_id"] for c in ranked], explanations
