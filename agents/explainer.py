from typing import Dict, Any, List
import json


class ExplainerAgent:
    """
    Explains a VERIFIED solution in a student-friendly manner.
    This agent must NEVER re-solve or modify the solution.
    """

    def __init__(self, llm, style: str = "friendly"):
        self.llm = llm
        self.style = style

    def _create_prompt(
        self,
        problem_text: str,
        verified_solution: Dict[str, Any],
        verification_confidence: float
    ) -> str:
        style_instructions = {
            "friendly": "Use a friendly, conversational tone.",
            "formal": "Use a formal, academic tone.",
            "concise": "Be brief but clear.",
            "detailed": "Provide thorough explanations with helpful context."
        }.get(self.style, "Use a clear and professional tone.")

        steps = "\n".join(
            f"{i+1}. {step}"
            for i, step in enumerate(verified_solution.get("steps", []))
        )

        return f"""
            You are an Explainer Agent.

            Your ONLY job is to explain the reasoning behind an ALREADY VERIFIED solution.

            ABSOLUTE RULES (NON-NEGOTIABLE):
            1. DO NOT re-solve the problem under any circumstance.
            2. DO NOT modify, correct, optimize, or reinterpret any step.
            3. DO NOT add new steps, assumptions, examples, or alternative approaches.
            4. DO NOT infer missing logic or "fix" unclear reasoning.
            5. DO NOT contradict the verified solution, even if it appears wrong.
            6. DO NOT introduce external knowledge beyond what is strictly required to explain WHY a step works.
            7. DO NOT restate the final answer in a different form.

            STRICT SCOPE:
            - You may ONLY explain the steps exactly as provided.
            - Each explanation must answer: "Why does this step logically follow?"
            - If a step is unclear or jumps logic, explain the underlying principle that makes it valid — NOT what should have been done instead.
            - If a step relies on a known concept, name the concept, but do not derive or expand it.

            TONE & STYLE:
            {style_instructions}

            CONTEXT GUARANTEES:
            - The solution has already been VERIFIED.
            - Verification confidence: {verification_confidence}
            - Treat the solution as ground truth.

            INPUTS:
            Problem:
            {problem_text}

            Verified Solution Steps:
            {steps}

            Final Answer (DO NOT MODIFY OR REPHRASE):
            {verified_solution.get("final_answer", "Not provided")}

            OUTPUT CONSTRAINTS:
            - Return VALID JSON ONLY.
            - No markdown.
            - No extra keys.
            - No prose outside JSON.
            - Each explanation must map ONE-TO-ONE with the provided steps.

            OUTPUT FORMAT (STRICT):
            {{
            "explanation": [
                "Why step 1 works...",
                "Why step 2 works..."
            ],
            "key_concepts": [
                "Concept explicitly used in the steps only"
            ],
            "common_mistakes": [
                "Mistakes people make when misunderstanding THESE steps (not alternative methods)"
            ]
            }}

            FAILURE MODE:
            If you cannot comply with ANY rule above, return:
            {{
            "explanation": [],
            "key_concepts": [],
            "common_mistakes": []
            }}

"""

    def explain(
        self,
        problem_text: str,
        verified_solution: Dict[str, Any],
        verification_confidence: float
    ) -> Dict[str, Any]:

        if verification_confidence < 0.85:
            raise ValueError(
                "ExplainerAgent should not run on low-confidence solutions"
            )

        prompt = self._create_prompt(
            problem_text,
            verified_solution,
            verification_confidence
        )

        response = self.llm.generate(prompt, temperature=0.3)

        try:
            if isinstance(response, str):
                parsed = json.loads(response)
            else:
                parsed = response

            explanation = parsed.get("explanation", [])
            key_concepts = parsed.get("key_concepts", [])
            common_mistakes = parsed.get("common_mistakes", [])

            # Normalize outputs
            if isinstance(explanation, str):
                explanation = [explanation]

            return {
                "explanation": explanation,
                "key_concepts": key_concepts,
                "common_mistakes": common_mistakes
            }

        except Exception:
            # Honest fallback — do not pretend completeness
            return {
                "explanation": [
                    "A complete explanation could not be generated reliably.",
                    "Below is a high-level walkthrough of the verified steps:"
                ] + [
                    f"- {step}" for step in verified_solution.get("steps", [])
                ],
                "key_concepts": [],
                "common_mistakes": [
                    "Detailed explanation unavailable due to generation error"
                ]
            }
