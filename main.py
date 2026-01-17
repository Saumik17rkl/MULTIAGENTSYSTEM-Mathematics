import os
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from agents.intent_router import IntentRouter
from agents.solver import SolverAgent
from agents.verifier import VerifierAgent
from agents.explainer import ExplainerAgent
from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient
from tools.python_tool import PythonTool

# --------------------------------------------------
# ENV + APP INIT
# --------------------------------------------------

load_dotenv()
app = Flask(__name__)

CONFIDENCE_THRESHOLD = 0.85


# --------------------------------------------------
# MULTI-AGENT SYSTEM
# --------------------------------------------------

class MultiAgentSystem:
    """
    Orchestrates the multi-agent math reasoning pipeline.
    Responsible ONLY for control flow and safety.
    """

    def __init__(self, llm_provider: str = "auto"):
        self.llm = self._initialize_llm(llm_provider)
        self.python_tool = PythonTool()

        self.intent_router = IntentRouter(self.llm)
        self.solver = SolverAgent(self.llm, self.python_tool)
        self.verifier = VerifierAgent(self.llm)
        self.explainer = ExplainerAgent(self.llm)

    def _initialize_llm(self, provider: str):
        provider = provider.lower()

        # Try Gemini first
        if provider in ("gemini", "auto"):
            try:
                gemini = GeminiClient()
                test = gemini.generate("ping")
                if test.get("success"):
                    print("✅ Using Gemini as primary LLM")
                    return gemini
                print("⚠️ Gemini failed, falling back to Groq")
            except Exception as e:
                print(f"⚠️ Gemini init error: {str(e)[:120]}")

        # Fallback to Groq
        groq = GroqClient(model_name="llama-3.3-70b-versatile")
        print("✅ Using Groq as fallback LLM")
        return groq

    def process_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_trace: List[Dict[str, Any]] = []

        try:
            # 1️⃣ Intent Routing
            route_info = self.intent_router.route(problem_data)
            agent_trace.append({"agent": "IntentRouter", "output": route_info})

            if route_info["route"] == "out_of_scope":
                return {
                    "status": "OUT_OF_SCOPE",
                    "reason": route_info.get("reason", "Unsupported problem"),
                    "agent_trace": agent_trace
                }

            # 2️⃣ Solve
            solution = self.solver.solve(
                problem_text=problem_data["problem_text"],
                route=route_info["route"],
                difficulty=route_info["difficulty"],
                tools_allowed=route_info.get("tools_allowed", []),
                rag_context=problem_data.get("retrieved_context", [])
            )
            agent_trace.append({"agent": "Solver", "output": solution})

            if solution.get("status") != "SOLVED":
                return {
                    "status": "FAILED",
                    "reason": solution.get("reason", "Solver failed"),
                    "agent_trace": agent_trace
                }

            # 3️⃣ Verify
            verification = self.verifier.verify(
                problem_text=problem_data["problem_text"],
                solution=solution,
                route=route_info["route"]
            )
            agent_trace.append({"agent": "Verifier", "output": verification})

            if (
                verification["verdict"] != "correct"
                or verification["confidence"] < CONFIDENCE_THRESHOLD
            ):
                return {
                    "status": "HITL_REQUIRED",
                    "reason": verification,
                    "proposed_solution": solution,
                    "agent_trace": agent_trace
                }

            # 4️⃣ Explain
            explanation = self.explainer.explain(
                problem_text=problem_data["problem_text"],
                verified_solution=solution,
                verification_confidence=verification["confidence"]
            )
            agent_trace.append({"agent": "Explainer", "output": explanation})

            return {
                "status": "SUCCESS",
                "final_answer": solution["final_answer"],
                "steps": solution["steps"],
                "explanation": explanation,
                "confidence": verification["confidence"],
                "agent_trace": agent_trace
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "agent_trace": agent_trace
            }


# --------------------------------------------------
# GLOBAL SYSTEM (INIT ONCE)
# --------------------------------------------------

system = MultiAgentSystem(llm_provider="auto")


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return """<pre>
Math Reasoning System
====================

A multi-agent system for solving mathematical problems.

Available Routes:
  GET  /health  - Check service health
  POST /solve   - Submit a math problem (requires JSON payload)
  
Example request:
  POST /solve
  {
    "problem_text": "Solve 2x + 5 = 15",
    "topic": "algebra",
    "variables": ["x"],
    "retrieved_context": []
  }
</pre>"""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "llm_provider": system.llm.__class__.__name__
    }), 200


@app.route("/solve", methods=["POST"])
def solve():
    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "Request must be JSON"
        }), 400

    data = request.get_json()

    if "problem_text" not in data or not isinstance(data["problem_text"], str):
        return jsonify({
            "status": "error",
            "error": "Field 'problem_text' (string) is required"
        }), 400

    problem_data = {
        "problem_text": data["problem_text"],
        "topic": data.get("topic", ""),
        "variables": data.get("variables", []),
        "constraints": data.get("constraints", []),
        "retrieved_context": data.get("retrieved_context", [])
    }

    result = system.process_problem(problem_data)
    return jsonify(result), 200


# --------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug)
