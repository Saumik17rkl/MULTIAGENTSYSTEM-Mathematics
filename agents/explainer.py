from typing import Dict, Any, List
import json


class ExplainerAgent:
    """
    Explains a VERIFIED and APPROVED solution.

    ABSOLUTE CONSTRAINTS:
    - Never re-solve
    - Never modify steps
    - Never introduce new reasoning
    - Never decide correctness
    """

    def __init__(self, llm, style: str = "friendly"):
        self.llm = llm
        self.style = style

    def _create_prompt(
        self,
        problem_text: str,
        verified_solution: Dict[str, Any]
    ) -> str:
        style_instructions = {
            "friendly": "Use a friendly, student-friendly tone.",
            "formal": "Use a formal, academic tone.",
            "concise": "Be brief and precise.",
            "detailed": "Explain carefully without adding new logic."
        }.get(self.style, "Use a clear and neutral tone.")

        steps: List[str] = verified_solution.get("steps", [])

        formatted_steps = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(steps)
        )

        return f"""
You are an Explainer Agent.

The solution below has ALREADY BEEN:
- verified by a Verifier Agent
- optionally approved or corrected by a human

You MUST treat it as ground truth.

────────────────────────────────────────
NON-NEGOTIABLE RULES
────────────────────────────────────────
1. DO NOT re-solve the problem.
2. DO NOT modify, reorder, or reinterpret steps.
3. DO NOT add assumptions, examples, or alternatives.
4. DO NOT fix unclear reasoning.
5. DO NOT contradict the provided steps.
6. DO NOT restate or reformat the final answer.
7. DO NOT introduce external knowledge unless explicitly required to explain WHY a step works.

────────────────────────────────────────
SCOPE OF EXPLANATION
────────────────────────────────────────
- Explain EACH step EXACTLY as provided.
- One explanation per step.
- Explanation answers ONLY: "Why does this step logically follow?"
- If a step relies on a known concept, name it briefly. Do NOT derive it.

────────────────────────────────────────
TONE
────────────────────────────────────────
{style_instructions}

────────────────────────────────────────
INPUT
────────────────────────────────────────
Problem:
{problem_text}

Verified Steps:
{formatted_steps}

Final Answer (READ-ONLY):
{verified_solution.get("final_answer", "")}

────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────
- Output VALID JSON only.
- No markdown.
- No extra keys.
- explanation.length MUST equal number of steps.
- If compliance is impossible, return EMPTY ARRAYS.

────────────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────────────
{{
  "explanation": [
    "Why step 1 works...",
    "Why step 2 works..."
  ],
  "key_concepts": [
    "Only concepts explicitly used in the steps"
  ],
  "common_mistakes": [
    "Misunderstandings related ONLY to these steps"
  ]
}}

FAILURE MODE:
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
        verification_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        NOTE:
        verification_confidence is accepted for logging only.
        It MUST NOT affect execution.
        """

        steps = verified_solution.get("steps", [])

        if not isinstance(steps, list):
            return {
                "explanation": [],
                "key_concepts": [],
                "common_mistakes": []
            }

        prompt = self._create_prompt(problem_text, verified_solution)
        response = self.llm.generate(prompt, temperature=0.2)

        try:
            parsed = (
                json.loads(response)
                if isinstance(response, str)
                else response
            )

            explanation = parsed.get("explanation", [])
            key_concepts = parsed.get("key_concepts", [])
            common_mistakes = parsed.get("common_mistakes", [])

            # HARD VALIDATION: 1-to-1 mapping
            if not isinstance(explanation, list) or len(explanation) != len(steps):
                return {
                    "explanation": [],
                    "key_concepts": [],
                    "common_mistakes": []
                }

            return {
                "explanation": explanation,
                "key_concepts": key_concepts if isinstance(key_concepts, list) else [],
                "common_mistakes": common_mistakes if isinstance(common_mistakes, list) else []
            }

        except Exception:
            # In HITL systems, silence > hallucination
            return {
                "explanation": [],
                "key_concepts": [],
                "common_mistakes": []
            }
