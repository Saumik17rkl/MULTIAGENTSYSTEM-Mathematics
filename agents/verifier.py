from typing import Dict, Any, List


class VerifierAgent:
    """
    Verifies correctness of solver output.
    This agent NEVER solves or fixes the problem.
    """

    def __init__(self, llm, confidence_threshold: float = 0.85):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def _create_prompt(
        self,
        problem_text: str,
        solution: Dict[str, Any],
        route: str
    ) -> str:

        steps_text = "\n".join(
            f"- {step}" for step in solution.get("steps", [])
        )

        return f"""
You are a Verifier Agent.

Your task is to evaluate whether the proposed solution is correct.

Rules:
- Do NOT solve the problem again.
- Do NOT suggest alternative solutions.
- Only evaluate correctness.
- Be strict but fair.

Problem Route:
{route}

Problem:
{problem_text}

Proposed Steps:
{steps_text}

Final Answer:
{solution.get("final_answer")}

Return ONLY valid JSON in this EXACT format:
{{
  "verdict": "correct or incorrect or uncertain",
  "confidence": 0.0,
  "issues": ["concise, step-referenced issues only"]
}}
"""

    def verify(
        self,
        problem_text: str,
        solution: Dict[str, Any],
        route: str
    ) -> Dict[str, Any]:

        # 🚨 Solver failure → no verifier escalation
        if solution.get("status") != "SOLVED":
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "issues": ["Solver did not produce a valid solution"],
                "needs_hitl": False
            }

        prompt = self._create_prompt(problem_text, solution, route)
        llm_response = self.llm.generate(prompt, temperature=0.1)

        # 🚨 LLM failure
        if not llm_response.get("success"):
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "issues": ["Verifier LLM call failed"],
                "needs_hitl": True
            }

        data = llm_response.get("parsed_json")

        if not isinstance(data, dict):
            return {
                "verdict": "uncertain",
                "confidence": 0.0,
                "issues": ["Verifier returned invalid JSON"],
                "needs_hitl": True
            }

        verdict = data.get("verdict", "uncertain")
        confidence = data.get("confidence", 0.0)
        issues = data.get("issues", [])

        # Normalize confidence
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        needs_hitl = (
            verdict != "correct"
            or confidence < self.confidence_threshold
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "issues": issues,
            "needs_hitl": needs_hitl
        }
