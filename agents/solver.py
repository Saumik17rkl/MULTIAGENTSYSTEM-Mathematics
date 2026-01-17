from typing import Dict, Any, List, Optional
import json
from tools.python_tool import PythonTool


class SolverAgent:
    """
    Solves a structured math problem using bounded reasoning.
    This agent MAY be wrong. The Verifier exists to judge correctness.
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

            Your sole responsibility is to solve the given problem and return a response
            that STRICTLY conforms to the required JSON schema.

            ────────────────────────────────────────
            STRICT OUTPUT GUARANTEES (NON-NEGOTIABLE)
            ────────────────────────────────────────
            1. Output MUST be valid, parseable JSON.
            2. Output MUST contain exactly these top-level keys:
            - "final_answer"
            - "steps"
            - "tool_requests"
            3. No extra keys. No missing keys. No comments.
            4. Do NOT include markdown, explanations, or text outside JSON.
            5. If any rule cannot be satisfied, return:
            {{
                "final_answer": "",
                "steps": [],
                "tool_requests": []
            }}

            ────────────────────────────────────────
            STEPS FIELD RULES (CRITICAL)
            ────────────────────────────────────────
            - "steps" MUST be a JSON ARRAY.
            - Each element MUST be a STRING.
            - Each string MUST describe ONE atomic reasoning step.
            - DO NOT combine multiple actions in one step.
            - DO NOT return steps as paragraphs or bullet-like text.
            - DO NOT expose hidden chain-of-thought or speculative reasoning.
            - If the reasoning cannot be safely decomposed into clear steps,
            return an EMPTY ARRAY.

            ────────────────────────────────────────
            FINAL ANSWER RULES
            ────────────────────────────────────────
            - "final_answer" MUST be a concise, direct solution.
            - No hedging, no speculation, no filler.
            - If the problem is unsolvable with given information, explicitly say so.

            ────────────────────────────────────────
            TOOL USAGE RULES
            ────────────────────────────────────────
            - Allowed tools: {tools_allowed}
            - If no tools are required, return an EMPTY ARRAY.
            - If tools are required:
            - Each request must be explicit and minimal.
            - Never hallucinate tool outputs.
            - Do NOT execute tools unless explicitly instructed.

            ────────────────────────────────────────
            INPUT METADATA (READ-ONLY)
            ────────────────────────────────────────
            Problem Route: {route}
            Difficulty: {difficulty}

            Problem:
            {problem_text}

            Retrieved Context:
            {context}

            ────────────────────────────────────────
            EXACT OUTPUT FORMAT (NO DEVIATION)
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

        # 🚨 Hard stop for out-of-scope
        if route == "out_of_scope":
            return {
                "status": "FAILED",
                "reason": "Problem out of supported scope",
                "final_answer": None,
                "steps": []
            }

        prompt = self._create_prompt(
            problem_text,
            route,
            difficulty,
            tools_allowed,
            rag_context
        )

        # 🔹 Call LLM (wrapper contract)
        llm_response = self.llm.generate(prompt, temperature=0.2)

        if not llm_response.get("success"):
            return {
                "status": "FAILED",
                "reason": llm_response.get("error", "LLM call failed"),
                "final_answer": None,
                "steps": []
            }

        solution = llm_response.get("parsed_json")

        if not isinstance(solution, dict):
            return {
                "status": "FAILED",
                "reason": "LLM did not return valid JSON",
                "final_answer": None,
                "steps": []
            }

        # ---- REQUIRED KEYS VALIDATION ----
        for key in ("final_answer", "steps", "tool_requests"):
            if key not in solution:
                return {
                    "status": "FAILED",
                    "reason": f"Missing required key: {key}",
                    "final_answer": None,
                    "steps": []
                }

        # ---- Normalize & Validate steps ----
        steps = solution["steps"]

        if isinstance(steps, str):
            steps = [
                s.strip("-• \n\t")
                for s in steps.split("\n")
                if s.strip()
            ]

        if not isinstance(steps, list):
            return {
                "status": "FAILED",
                "reason": "Steps must be a list",
                "final_answer": None,
                "steps": []
            }

        # ---- Validate final_answer ----
        final_answer = solution["final_answer"]
        if not isinstance(final_answer, str) or not final_answer.strip():
            return {
                "status": "FAILED",
                "reason": "Final answer is empty or invalid",
                "final_answer": None,
                "steps": []
            }

        # ---- Tool enforcement ----
        used_tools = []
        for tool in solution.get("tool_requests", []):
            if tool not in tools_allowed:
                return {
                    "status": "FAILED",
                    "reason": f"Unauthorized tool request: {tool}",
                    "final_answer": None,
                    "steps": []
                }
            used_tools.append(tool)

        return {
            "status": "SOLVED",
            "final_answer": final_answer,
            "steps": steps,
            "used_tools": used_tools
        }