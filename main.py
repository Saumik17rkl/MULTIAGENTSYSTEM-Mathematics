import os
import uuid
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
ALLOWED_HITL_ACTIONS = {"approve", "reject", "edit_problem", "correct_solution"}
TERMINAL_STATES = {"RESOLVED"}

# Temporary in-memory HITL store
HITL_STORE: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------
# MULTI-AGENT SYSTEM
# --------------------------------------------------

class MultiAgentSystem:

    def __init__(self, llm_provider: str = "auto"):
        self.llm = self._initialize_llm(llm_provider)
        self.python_tool = PythonTool()

        self.intent_router = IntentRouter(self.llm)
        self.solver = SolverAgent(self.llm, self.python_tool)
        self.verifier = VerifierAgent(self.llm)
        self.explainer = ExplainerAgent(self.llm)

    def _initialize_llm(self, provider: str):
        provider = provider.lower()
        if provider in ("gemini", "auto"):
            try:
                gemini = GeminiClient()
                if gemini.generate("ping").get("success"):
                    return gemini
            except Exception:
                pass
        return GroqClient(model_name="llama-3.3-70b-versatile")

    # --------------------------------------------------
    # PRIMARY PIPELINE
    # --------------------------------------------------

    def process_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_trace: List[Dict[str, Any]] = []

        route_info = self.intent_router.route(problem_data)
        agent_trace.append({"agent": "IntentRouter", "output": route_info})

        if route_info["route"] == "out_of_scope":
            return {"status": "OUT_OF_SCOPE", "agent_trace": agent_trace}

        solver_result = self.solver.solve(
            problem_text=problem_data["problem_text"],
            route=route_info["route"],
            difficulty=route_info["difficulty"],
            tools_allowed=route_info.get("tools_allowed", []),
            rag_context=problem_data.get("retrieved_context", [])
        )
        agent_trace.append({"agent": "Solver", "output": solver_result})

        if solver_result.get("status") != "SOLVED":
            return {"status": "FAILED", "agent_trace": agent_trace}

        verification = self.verifier.verify(
            problem_text=problem_data["problem_text"],
            solution=solver_result,
            route=route_info["route"]
        )
        agent_trace.append({"agent": "Verifier", "output": verification})

        if (
            verification.get("verdict") != "correct"
            or verification.get("confidence", 0.0) < CONFIDENCE_THRESHOLD
            or verification.get("requires_hitl", False)
        ):
            hitl_id = str(uuid.uuid4())

            HITL_STORE[hitl_id] = {
                "state": "PENDING_REVIEW",
                "problem_data": problem_data,
                "solution": solver_result,
                "verification": verification,
                "agent_trace": agent_trace,
                "hitl_reason": {
                    "verdict": verification.get("verdict"),
                    "confidence": verification.get("confidence"),
                    "requires_hitl": verification.get("requires_hitl", False)
                }
            }

            return {
                "status": "HITL_REQUIRED",
                "hitl_request_id": hitl_id,
                "hitl_reason": HITL_STORE[hitl_id]["hitl_reason"]
            }

        explanation = self.explainer.explain(
            problem_text=problem_data["problem_text"],
            verified_solution=solver_result,
            verification_confidence=verification["confidence"]
        )

        return {
            "status": "SUCCESS",
            "final_answer": solver_result["final_answer"],
            "steps": solver_result["steps"],
            "explanation": explanation,
            "confidence": verification["confidence"],
            "agent_trace": agent_trace
        }

    # --------------------------------------------------
    # HITL RESUME
    # --------------------------------------------------

    def resume_from_hitl(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        hitl_id = payload["hitl_request_id"]
        action = payload["action"]

        if action not in ALLOWED_HITL_ACTIONS:
            return {"status": "ERROR", "error": "Invalid HITL action"}

        record = HITL_STORE.get(hitl_id)

        if not record:
            return {"status": "ERROR", "error": "Invalid HITL request ID"}

        if record["state"] in TERMINAL_STATES:
            return {
                "status": "ERROR",
                "error": f"HITL request already resolved (state={record['state']})"
            }

        if action == "approve":
            explanation = self.explainer.explain(
                problem_text=record["problem_data"]["problem_text"],
                verified_solution=record["solution"],
                verification_confidence=record["verification"]["confidence"]
            )
            record["state"] = "RESOLVED"
            return {
                "status": "SUCCESS",
                "final_answer": record["solution"]["final_answer"],
                "steps": record["solution"]["steps"],
                "explanation": explanation,
                "confidence": record["verification"]["confidence"]
            }

        if action == "reject":
            record["state"] = "RESOLVED"
            return {"status": "REJECTED"}

        if action == "edit_problem":
            edited_text = payload.get("edited_problem_text")
            if not edited_text:
                return {"status": "ERROR", "error": "edited_problem_text required"}

            record["problem_data"]["problem_text"] = edited_text
            record["state"] = "RESOLVED"
            record["resolution_type"] = "EDITED_PROBLEM"

            return {
                "status": "NEEDS_RERUN",
                "updated_problem_data": record["problem_data"],
                "instruction": "Re-run pipeline from IntentRouter"
            }


        if action == "correct_solution":
            corrected = payload.get("corrected_solution")
            if not corrected:
                return {"status": "ERROR", "error": "corrected_solution required"}

            required_fields = {"final_answer", "steps"}
            if not required_fields.issubset(corrected):
                return {
                    "status": "ERROR",
                    "error": "corrected_solution must include final_answer and steps"
                }

            explanation = self.explainer.explain(
                problem_text=record["problem_data"]["problem_text"],
                verified_solution=corrected,
                verification_confidence=1.0
            )
            record["state"] = "RESOLVED"
            record["resolution_type"] = "EDITED_PROBLEM"

            return {
                "status": "SUCCESS",
                "final_answer": corrected.get("final_answer"),
                "steps": corrected.get("steps"),
                "explanation": explanation,
                "confidence": 1.0
            }

# --------------------------------------------------
# GLOBAL SYSTEM
# --------------------------------------------------
def cleanup_hitl_store():
    """
    Placeholder for TTL / DB cleanup.
    In production, resolved HITL records should expire.
    """
    pass

system = MultiAgentSystem(llm_provider="auto")

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/", methods=["GET"]) 
def index():
    return """<pre> Math Reasoning System ==================== A multi-agent system for solving mathematical problems. 
    Available Routes: GET /health - Check service health 
    POST /solve - Submit a math problem (requires JSON payload) 
    Example request: POST /solve { "problem_text": "Solve 2x + 5 = 15", "topic": "algebra", "variables": ["x"], "retrieved_context": [] } </pre>"""

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/solve", methods=["POST"])
def solve():
    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400

    data = request.get_json()

    if "problem_text" not in data:
        return jsonify({"error": "problem_text is required"}), 400

    return jsonify(system.process_problem(data)), 200


@app.route("/hitl/resolve", methods=["POST"])
def hitl_resolve():
    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400

    payload = request.get_json()

    if not {"hitl_request_id", "action"}.issubset(payload):
        return jsonify({"error": "Missing HITL fields"}), 400

    return jsonify(system.resume_from_hitl(payload)), 200



# --------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
