from typing import Dict, Any, List
import json


class IntentRouter:
    """
    Classifies a parsed math problem into a solver route.

    RULES:
    - Never solve
    - Never explain
    - Never transform the problem
    - Only classify intent
    """

    VALID_ROUTES = {
        "algebra_equation",
        "probability_basic",
        "calculus_limit",
        "calculus_derivative",
        "calculus_optimization",
        "linear_algebra_basic",
        "out_of_scope"
    }

    VALID_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}

    def __init__(self, llm):
        self.llm = llm

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    def _create_prompt(self, problem_data: Dict[str, Any]) -> str:
        return f"""
You are an Intent Router Agent.

Your ONLY task is to classify the problem into ONE predefined route.
You must NOT solve, explain, rewrite, or analyze the problem.

ALLOWED ROUTES (EXACT MATCH ONLY):
- algebra_equation
- probability_basic
- calculus_limit
- calculus_derivative
- calculus_optimization
- linear_algebra_basic
- out_of_scope

CLASSIFICATION RULES:
1. Choose the single dominant mathematical intent.
2. If intent is unclear, mixed, or outside scope → out_of_scope.
3. Difficulty is a rough estimate:
   - easy / medium / hard
   - use unknown if unsure.

FORBIDDEN:
- No explanations
- No reasoning
- No extra keys
- No text outside JSON

OUTPUT FORMAT (STRICT JSON):
{{
  "route": "<allowed_route>",
  "difficulty": "<easy|medium|hard|unknown>",
  "tools_allowed": []
}}

INPUT:
Problem:
{problem_data.get("problem_text", "")}

Topic (from parser):
{problem_data.get("topic", "unknown")}

Variables:
{problem_data.get("variables", [])}

Constraints:
{problem_data.get("constraints", [])}
"""

    # --------------------------------------------------
    # PRIMARY ROUTING LOGIC
    # --------------------------------------------------

    def route(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routing priority:
        1. Trust Parser Agent if topic is clean
        2. Fallback to LLM classification
        3. Fail closed (out_of_scope)
        """

        topic = problem_data.get("topic")

        TOPIC_TO_ROUTE = {
            "algebra": "algebra_equation",
            "probability": "probability_basic",
            "calculus_limit": "calculus_limit",
            "calculus_derivative": "calculus_derivative",
            "calculus_optimization": "calculus_optimization",
            "linear_algebra": "linear_algebra_basic"
        }

        # ✅ PRIMARY: Parser Agent authority
        if isinstance(topic, str) and topic in TOPIC_TO_ROUTE:
            return {
                "route": TOPIC_TO_ROUTE[topic],
                "difficulty": "medium",
                "tools_allowed": []
            }

        # 🔁 FALLBACK: LLM-based routing
        return self._llm_
