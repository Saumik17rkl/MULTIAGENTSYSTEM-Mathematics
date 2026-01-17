from typing import Dict, Any
import os
import json
from dotenv import load_dotenv
from google import genai


class GeminiClient:
    """
    Stable Gemini client using google.genai (new SDK).
    Contract:
    {
      success: bool,
      content: str,
      parsed_json: dict | None,
      error: str | None
    }
    """

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": 2048
                }
            )

            text = response.text.strip() if response.text else ""

            parsed_json = None
            try:
                parsed_json = json.loads(text)
            except json.JSONDecodeError:
                pass

            return {
                "success": True,
                "content": text,
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
