from typing import Any, Dict, List, Optional, Union
import sympy


class PythonTool:
    """
    Deterministic symbolic math tool.
    This tool MUST NOT mutate solver outputs silently.
    """

    def evaluate(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate or simplify a mathematical expression.

        Returns:
        {
          "success": bool,
          "result": float | str | None,
          "error": str | None
        }
        """
        try:
            expr = sympy.sympify(expression.replace("^", "**"))

            # Try numeric evaluation
            try:
                numeric = expr.evalf()
                if numeric.is_real:
                    return {
                        "success": True,
                        "result": float(numeric),
                        "error": None
                    }
            except Exception:
                pass

            # Fallback to symbolic simplification
            return {
                "success": True,
                "result": str(sympy.simplify(expr)),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }

    def solve_equation(
        self,
        equation: str,
        variable: str
    ) -> Dict[str, Any]:
        """
        Solve an equation for a variable.
        """
        try:
            if "=" in equation:
                lhs, rhs = equation.split("=")
                expr = sympy.sympify(f"({lhs}) - ({rhs})")
            else:
                expr = sympy.sympify(equation)

            var = sympy.Symbol(variable)
            solutions = sympy.solve(expr, var)

            return {
                "success": True,
                "result": [str(sol) for sol in solutions],
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }

    def derivative(
        self,
        expression: str,
        variable: str,
        order: int = 1
    ) -> Dict[str, Any]:
        """
        Compute the derivative.
        """
        try:
            x = sympy.Symbol(variable)
            expr = sympy.sympify(expression)
            result = sympy.diff(expr, x, order)

            return {
                "success": True,
                "result": str(result),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }

    def integral(
        self,
        expression: str,
        variable: str,
        lower: Optional[Any] = None,
        upper: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Compute an integral (definite or indefinite).
        """
        try:
            x = sympy.Symbol(variable)
            expr = sympy.sympify(expression)

            if lower is not None and upper is not None:
                result = sympy.integrate(expr, (x, lower, upper))
            else:
                result = sympy.integrate(expr, x)

            return {
                "success": True,
                "result": str(result),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
