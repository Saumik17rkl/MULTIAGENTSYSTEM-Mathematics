from typing import Dict, Any, List


class VerifierAgent:
    """
    Evaluates correctness of a candidate solution.

    AUTHORITY:
    - Judges correctness
    - Estimates confidence
    - Flags uncertainty

    RESTRICTIONS:
    - NEVER solve
    - NEVER fix
    - NEVER rewrite steps
    """

    def __init__(self, llm, confidence_threshold: float = 0.85):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    def _create_prompt(
        self,
        problem_text: str,
        solution: Dict[str, Any],
        route: str
    ) -> str:

        steps = solution.get("steps", [])

        steps_text = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(steps)
        )

        return f"""
You are a Verifier Agent.

Your ONLY task is to evaluate whether the proposed solution is correct.

STRICT RULES:
- DO NOT solve the problem.
- DO NOT suggest fixes or alternatives.
- DO NOT add steps.
- DO NOT rewrite reasoning.
- Only judge correctness and uncertainty.

EVALUATION CRITERIA:
- Logical correctness of each step
- Correct use of definitions, formulas, and constraints
- Consistency between steps and final answer
- Domain validity (no illegal operations)

PROBLEM ROUTE:
{route}

PROBLEM:
{problem_text}

PROPOSED STEPS:
{steps_text}

FINAL ANSWER:
{solution.get("final_answer", "")}

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "verdict": "correct | incorrect | uncertain",
  "confidence": 0.0,
  "issues": [
    "Step-referenced, concise issues only"
  ]
}}

IMPORTANT:
- confidence MUST be between 0.0 and 1.0
- If unsure, use verdict = "uncertain"
- If incorrect, list the exact step(s) involved
- No text outside JSON
"""

    # --------------------------------------------------
    # VERIFICATION
    # --------------------------------------------------

    def verify(
        self,
        problem_text: str,
        solution: Dict[str, Any],
        route: str
    ) -> Dict[str, Any]:

        # 🚨 Hard validation: solution structure
        if not isinstance(solution, dict):
            return self._fail_closed("Invalid solution structure")

        if not isinstance(solution.get("final_answer"), str):
            return self._fail_closed("Missing or invalid final_answer")

        if not isinstance(solution.get("steps"), list):
            return self._fail_closed("Missing or invalid steps")

        prompt = self._create_prompt(problem_text, solution, route)
        llm_response = self.llm.generate(prompt, temperature=0.1)

        # 🚨 LLM failure → require HITL
        if not llm_response.get("success"):
            return self._fail_closed("Verifier LLM call failed")

        data = llm_response.get("parsed_json")

        if not isinstance(data, dict):
            return self._fail_closed("Verifier returned invalid JSON")

        verdict = data.get("verdict", "uncertain")
        confidence = data.get("confidence", 0.0)
        issues = data.get("issues", [])

        # Normalize confidence
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        # Normalize verdict
        if verdict not in {"correct", "incorrect", "uncertain"}:
            verdict = "uncertain"

        # Normalize issues
        if not isinstance(issues, list):
            issues = ["Invalid issues format returned by verifier"]

        requires_hitl = (
            verdict != "correct"
            or confidence < self.confidence_threshold
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "issues": issues,
            "requires_hitl": requires_hitl
        }

    # --------------------------------------------------
    # FAIL CLOSED
    # --------------------------------------------------

    def _fail_closed(self, reason: str) -> Dict[str, Any]:
        """
        Any verifier failure MUST require HITL.
        """
        return {
            "verdict": "uncertain",
            "confidence": 0.0,
            "issues": [reason],
            "requires_hitl": True
        }
