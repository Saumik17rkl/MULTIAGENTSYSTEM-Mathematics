from typing import Dict, Any
import os
import json
from groq import Groq
from dotenv import load_dotenv


class GroqClient:
    """
    Stable LLM wrapper for Groq-hosted models.
    Contract:
    {
      success: bool,
      content: str,
      parsed_json: dict | None,
      error: str | None
    }
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise EnvironmentError("GROQ_API_KEY environment variable is not set")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=2048,
                top_p=0.95,
            )

            raw_text = ""
            if completion.choices and completion.choices[0].message:
                raw_text = completion.choices[0].message.content or ""

            raw_text = raw_text.strip()

            parsed_json = None
            try:
                parsed_json = json.loads(raw_text)
            except json.JSONDecodeError:
                pass  # Expected for many prompts

            return {
                "success": True,
                "content": raw_text,
                "parsed_json": parsed_json,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "content": "",
                "parsed_json": None,
                "error": str(e)
            }
