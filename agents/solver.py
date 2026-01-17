from typing import Dict, Any, List, Optional
from tools.python_tool import PythonTool


class SolverAgent:
    """
    Produces a candidate solution for a structured math problem.

    IMPORTANT:
    - This agent does NOT decide correctness.
    - This agent does NOT finalize answers.
    - All outputs are subject to verification and human review.
    """

    def __init__(self, llm, python_tool: Optional[PythonTool] = None):
        self.llm = llm
        self.python_tool = python_tool or PythonTool()

    def _create_prompt(
        self,
        problem_text: str,
        route: str,
        difficulty: str,
        tools_allowed: List[str],
        rag_context: Optional[List[str]]
    ) -> str:

        context = "\n".join(rag_context) if rag_context else "No external context provided."

        return f"""
You are a deterministic Solver Agent.

Your responsibility is to generate a *candidate solution*.
You are allowed to be wrong. A Verifier will judge correctness.

────────────────────────────────────────
STRICT OUTPUT GUARANTEES (NON-NEGOTIABLE)
────────────────────────────────────────
1. Output MUST be valid JSON.
2. Output MUST contain EXACTLY these keys:
   - "final_answer"
   - "steps"
   - "tool_requests"
3. NO extra keys. NO missing keys.
4. NO markdown. NO commentary. NO text outside JSON.
5. If rules cannot be satisfied, return:
{{
  "final_answer": "",
  "steps": [],
  "tool_requests": []
}}

────────────────────────────────────────
STEPS FIELD RULES
────────────────────────────────────────
- "steps" MUST be a JSON array of strings.
- Each string = ONE atomic step.
- No combined actions.
- No hidden chain-of-thought.
- If unclear, return an EMPTY array.

────────────────────────────────────────
FINAL ANSWER RULES
────────────────────────────────────────
- Must be concise and direct.
- No hedging.
- If unsolvable, explicitly state that.

────────────────────────────────────────
TOOL USAGE RULES
────────────────────────────────────────
- Allowed tools: {tools_allowed}
- If no tools are required → EMPTY array.
- Never fabricate tool outputs.

────────────────────────────────────────
INPUT METADATA (READ-ONLY)
────────────────────────────────────────
Route: {route}
Difficulty: {difficulty}

Problem:
{problem_text}

Retrieved Context:
{context}

────────────────────────────────────────
EXACT OUTPUT FORMAT
────────────────────────────────────────
{{
  "final_answer": "string",
  "steps": ["step 1", "step 2"],
  "tool_requests": []
}}
"""

    def solve(
        self,
        problem_text: str,
        route: str,
        difficulty: str,
        tools_allowed: List[str],
        rag_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        # Hard stop: unsupported domain
        if route == "out_of_scope":
            return {
                "status": "CANDIDATE_FAILED",
                "error": "Problem out of supported scope",
                "solution": None
            }

        prompt = self._create_prompt(
            problem_text,
            route,
            difficulty,
            tools_allowed,
            rag_context
        )

        llm_response = self.llm.generate(prompt, temperature=0.2)

        if not llm_response.get("success"):
            return {
                "status": "CANDIDATE_FAILED",
                "error": llm_response.get("error", "LLM call failed"),
                "solution": None
            }

        solution = llm_response.get("parsed_json")

        if not isinstance(solution, dict):
            return {
                "status": "CANDIDATE_FAILED",
                "error": "LLM did not return valid JSON",
                "solution": None
            }

        # Required keys check
        for key in ("final_answer", "steps", "tool_requests"):
            if key not in solution:
                return {
                    "status": "CANDIDATE_FAILED",
                    "error": f"Missing required key: {key}",
                    "solution": None
                }

        # Normalize steps
        steps = solution["steps"]
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.split("\n") if s.strip()]

        if not isinstance(steps, list):
            return {
                "status": "CANDIDATE_FAILED",
                "error": "Steps must be a list",
                "solution": None
            }

        final_answer = solution["final_answer"]
        if not isinstance(final_answer, str):
            return {
                "status": "CANDIDATE_FAILED",
                "error": "Final answer must be a string",
                "solution": None
            }

        # Tool validation
        used_tools = []
        for tool in solution.get("tool_requests", []):
            if tool not in tools_allowed:
                return {
                    "status": "CANDIDATE_FAILED",
                    "error": f"Unauthorized tool request: {tool}",
                    "solution": None
                }
            used_tools.append(tool)

        return {
            "status": "CANDIDATE_GENERATED",
            "solution": {
                "final_answer": final_answer.strip(),
                "steps": steps,
                "used_tools": used_tools
            }
        }
