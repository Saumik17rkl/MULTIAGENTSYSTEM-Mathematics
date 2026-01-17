from typing import Dict, Any, List
import json


class IntentRouter:
    """
    Classifies a parsed math problem into a solver route.
    This agent MUST NOT solve or explain.
    """

    VALID_ROUTES = {
        "algebra_equation",
        "probability_basic",
        "calculus_limit",
        "calculus_derivative",
        "calculus_optimization",
        "linear_algebra_basic"
    }

    VALID_DIFFICULTIES = {"easy", "medium", "hard"}

    VALID_TOOLS = {"python"}

    def __init__(self, llm):
        self.llm = llm

    def _create_prompt(self, problem_data: Dict[str, Any]) -> str:
        return f"""
You are an Intent Router Agent.

Your ONLY responsibility is to classify the given problem into ONE predefined route.
You are NOT allowed to solve, analyze, transform, rephrase, or explain the problem.

────────────────────
ALLOWED ROUTES (EXACT MATCH ONLY)
────────────────────
- algebra_equation
- probability_basic
- calculus_limit
- calculus_derivative
- calculus_optimization
- linear_algebra_basic

DO NOT create new routes.
DO NOT combine routes.
DO NOT infer advanced topics beyond the listed routes.

────────────────────
CLASSIFICATION RULES
────────────────────
1. Choose the SINGLE most dominant mathematical intent.
2. If multiple topics appear, select the PRIMARY objective, not supporting concepts.
3. If the problem requires:
   - solving equations → algebra_equation
   - basic probability rules, counting, conditional probability → probability_basic
   - evaluating limits → calculus_limit
   - finding derivatives (explicit or implicit) → calculus_derivative
   - maximizing/minimizing quantities → calculus_optimization
   - vectors, matrices, determinants, systems, eigen concepts → linear_algebra_basic
4. If the intent is unclear, mixed beyond scope, non-mathematical, or outside these domains → out_of_scope.

────────────────────
FORBIDDEN BEHAVIOR
────────────────────
- DO NOT solve the problem.
- DO NOT explain your choice.
- DO NOT add commentary or reasoning.
- DO NOT modify or reinterpret the problem.
- DO NOT output anything other than valid JSON.

────────────────────
OUTPUT FORMAT (STRICT)
────────────────────
Return ONLY valid JSON in this exact structure:

{
  "route": "<one_of_the_allowed_routes_or_out_of_scope>",
  "difficulty": "<easy | medium | hard | unknown>",
  "tools_allowed": []
}

────────────────────
INPUT
────────────────────
Problem:
{problem_data.get("problem_text", "")}

Topic:
{problem_data.get("topic", "unknown")}

Variables:
{problem_data.get("variables", [])}

Constraints:
{problem_data.get("constraints", [])}

JSON:

"""

    def route(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route based on Parser Agent output.
        LLM routing is fallback only.
        """

        topic = problem_data.get("topic")

        TOPIC_TO_ROUTE = {
            "algebra": "algebra_equation",
            "probability": "probability_basic",
            "calculus": "calculus_derivative",
            "linear_algebra": "linear_algebra_basic"
        }

        # ✅ PRIMARY: TRUST PARSER AGENT
        if topic in TOPIC_TO_ROUTE:
            return {
                "route": TOPIC_TO_ROUTE[topic],
                "difficulty": "medium",
                "tools_allowed": ["python"]
            }

        # ❌ ONLY IF PARSER FAILED → LLM ROUTING
        return self._llm_route(problem_data)

    def _out_of_scope(self, reason: str) -> Dict[str, Any]:
        return {
            "route": "out_of_scope",
            "difficulty": "unknown",
            "tools_allowed": [],
            "reason": reason
        }
