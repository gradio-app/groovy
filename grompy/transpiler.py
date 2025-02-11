from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from collections.abc import Callable


class TranspilerError(Exception):
    """Exception raised when transpilation fails or encounters ambiguous syntax."""

    def __init__(
        self,
        issues: list[tuple[int, str, str]] | None = None,
        message: str | None = None,
    ):
        self.issues = issues or []
        if message:
            super().__init__(message)
        else:
            issue_count = len(self.issues)
            issues_text = (
                f"{issue_count} issue{'s' if issue_count != 1 else ''} found:\n\n"
            )
            for line_no, message, code in self.issues:
                issues_text += f"* Line {line_no}: {message}\n>> {code}\n"
            super().__init__(issues_text)


class PythonToJSVisitor(ast.NodeVisitor):
    def __init__(self):
        self.js_lines = []  # Accumulate lines of JavaScript code.
        self.indent_level = 0  # Track current indent level for readability.
        self.declared_vars = set()  # Track declared variables
        self.issues: list[tuple[int, str, str]] = []  # Track transpilation issues
        self.source_lines: list[str] = []  # Store source code lines
        self.var_types: dict[str, type | None] = {}  # Track variable types

    def add_issue(self, node: ast.AST, message: str) -> None:
        """Add a transpilation issue with source code context."""
        if hasattr(node, "lineno"):
            line_no = node.lineno
            line_text = self.source_lines[line_no - 1].strip()
            self.issues.append((line_no, message, line_text))

    def visit(self, node):
        try:
            return super().visit(node)
        except TranspilerError as e:
            if e.issues:
                self.issues.extend(e.issues)
            return ""  # Return empty string for failed conversions

    def generic_visit(self, node):
        self.add_issue(node, f"Unsupported syntax: {type(node).__name__}")
        return ""

    def indent(self) -> str:
        return "    " * self.indent_level

    # === Function Definition ===
    def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
        # Extract parameter names and types from type hints
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                type_name = arg.annotation.id
                type_obj = globals().get(type_name, getattr(builtins, type_name, None))
                if type_obj is not None:
                    self.var_types[arg.arg] = type_obj

        header = f"function {node.name}({', '.join(params)}) " + "{"
        self.js_lines.append(header)
        self.indent_level += 1

        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.js_lines.append("}")

    # === Return Statement ===
    def visit_Return(self, node: ast.Return):  # noqa: N802
        ret_val = "" if node.value is None else self.visit(node.value)
        self.js_lines.append(f"{self.indent()}return {ret_val};")

    # === Expression Statements ===
    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        expr = self.visit(node.value)
        self.js_lines.append(f"{self.indent()}{expr};")

    # === Assignment ===
    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        if len(node.targets) != 1:
            raise TranspilerError("Multiple assignment targets are not supported yet.")
        target_node = node.targets[0]
        target = self.visit(target_node)
        value = self.visit(node.value)

        if (
            isinstance(target_node, ast.Name)
            and target_node.id not in self.declared_vars
        ):
            self.declared_vars.add(target_node.id)
            expr_type = self.get_expr_type(node.value)
            if expr_type is not None:
                self.var_types[target_node.id] = expr_type
            self.js_lines.append(f"{self.indent()}let {target} = {value};")
        else:
            self.js_lines.append(f"{self.indent()}{target} = {value};")

    # === Binary Operations ===
    def check_type_safety(self, node: ast.AST, *exprs: ast.AST, context: str) -> None:
        """
        Check if an operation is type-safe.
        Raises TranspilerError if types are ambiguous or incompatible.
        """
        types = [self.get_expr_type(expr) for expr in exprs]

        # Check for unknown types
        if any(t is None for t in types):
            self.add_issue(
                node,
                f"Ambiguous operation: Cannot determine types for {context}. "
                "Operation behavior may differ in JavaScript based on types.",
            )
            raise TranspilerError()

        # Check for type consistency if multiple expressions
        if len(types) > 1 and not all(t == types[0] for t in types):
            type_names = [t.__name__ for t in types]
            self.add_issue(
                node,
                f"Ambiguous operation: Mixed types ({', '.join(type_names)}) in {context}. "
                "Behavior may differ in JavaScript.",
            )
            raise TranspilerError()

    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.visit(node.op)

        self.check_type_safety(
            node, node.left, node.right, context=f"'{left} {op} {right}'"
        )
        return f"({left} {op} {right})"

    def visit_Add(self, node: ast.Add):  # noqa: N802, ARG002
        return "+"

    def visit_Sub(self, node: ast.Sub):  # noqa: N802, ARG002
        return "-"

    def visit_Mult(self, node: ast.Mult):  # noqa: N802, ARG002
        return "*"

    def visit_Div(self, node: ast.Div):  # noqa: N802, ARG002
        return "/"

    # === Comparison Operations ===
    def visit_Compare(self, node: ast.Compare):  # noqa: N802
        left = self.visit(node.left)
        ops = [self.visit(op) for op in node.ops]
        comparators = [self.visit(comp) for comp in node.comparators]

        # Check types for comparison operations
        exprs = [node.left] + node.comparators
        self.check_type_safety(
            node,
            *exprs,
            context=f"comparison {left} {' '.join(ops)} {' '.join(comparators)}",
        )

        if len(ops) != 1 or len(comparators) != 1:
            raise TranspilerError("Only single comparisons are supported")

        return f"({left} {ops[0]} {comparators[0]})"

    def visit_Gt(self, node: ast.Gt):  # noqa: N802, ARG002
        return ">"

    def visit_Lt(self, node: ast.Lt):  # noqa: N802, ARG002
        return "<"

    def visit_GtE(self, node: ast.GtE):  # noqa: N802, ARG002
        return ">="

    def visit_LtE(self, node: ast.LtE):  # noqa: N802, ARG002
        return "<="

    def visit_Eq(self, node: ast.Eq):  # noqa: N802, ARG002
        return "==="

    def visit_NotEq(self, node: ast.NotEq):  # noqa: N802, ARG002
        return "!=="

    # === If Statement ===
    def visit_If(self, node: ast.If):  # noqa: N802
        test = self.visit(node.test)
        self.js_lines.append(f"{self.indent()}if ({test}) " + "{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.js_lines.append(f"{self.indent()}" + "}")

        # Handle elif and else clauses
        current = node
        while (
            current.orelse
            and len(current.orelse) == 1
            and isinstance(current.orelse[0], ast.If)
        ):
            current = current.orelse[0]
            test = self.visit(current.test)
            self.js_lines.append(f"{self.indent()}else if ({test}) " + "{")
            self.indent_level += 1
            for stmt in current.body:
                self.visit(stmt)
            self.indent_level -= 1
            self.js_lines.append(f"{self.indent()}" + "}")

        # Handle final else clause if it exists
        if current.orelse:
            self.js_lines.append(f"{self.indent()}else " + "{")
            self.indent_level += 1
            for stmt in current.orelse:
                self.visit(stmt)
            self.indent_level -= 1
            self.js_lines.append(f"{self.indent()}" + "}")

    # === Function Calls ===
    def visit_range(self, args: list[str]) -> str:
        """Convert Python's range() to equivalent JavaScript array."""
        if len(args) == 1:  # range(stop)
            return f"Array.from({{length: {args[0]}}}, (_, i) => i)"
        elif len(args) == 2:  # range(start, stop)
            return f"Array.from({{length: {args[1]} - {args[0]}}}, (_, i) => i + {args[0]})"
        elif len(args) == 3:  # range(start, stop, step)
            raise TranspilerError("range() with step argument is not supported yet")
        else:
            raise TranspilerError("Invalid number of arguments for range()")

    def visit_Call(self, node: ast.Call):  # noqa: N802
        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                args = [self.visit(arg) for arg in node.args]
                for arg in node.args:
                    self.check_type_safety(arg, arg, context="range() argument")
                return self.visit_range(args)

            for arg in node.args:
                self.check_type_safety(
                    arg, arg, context=f"argument in {node.func.id}() call"
                )

            self.add_issue(node, f'Unsupported function "{node.func.id}()"')
            return ""

        # For method calls (like obj.method())
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        # Check types for method calls
        if isinstance(node.func, ast.Attribute):
            self.check_type_safety(
                node.func, node.func.value, context=f"object in method call {func}"
            )
            for arg in node.args:
                self.check_type_safety(
                    arg, arg, context=f"argument in method call {func}"
                )

        return f"{func}({', '.join(args)})"

    # === Variable Name ===
    def visit_Name(self, node: ast.Name):  # noqa: N802
        return node.id

    # === Constants ===
    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if node.value is None:
            return "null"
        return repr(node.value)

    # === For Loop ===
    def visit_For(self, node: ast.For):  # noqa: N802
        target = self.visit(node.target)
        iter_expr = self.visit(node.iter)

        # If iterating over a range, record that the loop variable is an int.
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            if isinstance(node.target, ast.Name):
                self.var_types[node.target.id] = int

        self.js_lines.append(f"{self.indent()}for (let {target} of {iter_expr}) " + "{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.js_lines.append(f"{self.indent()}" + "}")

    # === While Loop ===
    def visit_While(self, node: ast.While):  # noqa: N802
        test = self.visit(node.test)
        self.js_lines.append(f"{self.indent()}while ({test}) " + "{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.js_lines.append(f"{self.indent()}" + "}")

    # === List ===
    def visit_List(self, node: ast.List):  # noqa: N802
        elements = [self.visit(elt) for elt in node.elts]
        return f"[{', '.join(elements)}]"

    # === Subscript ===
    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        value = self.visit(node.value)
        slice_value = self.visit(node.slice)
        return f"{value}[{slice_value}]"

    # === Augmented Assignment ===
    def visit_AugAssign(self, node: ast.AugAssign):  # noqa: N802
        target = self.visit(node.target)
        op = self.visit(node.op).strip()
        value = self.visit(node.value)
        self.js_lines.append(f"{self.indent()}{target} {op}= {value};")

    # === Boolean Operations ===
    def visit_BoolOp(self, node: ast.BoolOp):  # noqa: N802
        op = self.visit(node.op)
        values = [self.visit(value) for value in node.values]
        return f"({' ' + op + ' '.join(values)})"

    def visit_And(self, node: ast.And):  # noqa: N802, ARG002
        return "&&"

    def visit_Or(self, node: ast.Or):  # noqa: N802, ARG002
        return "||"

    # === Dictionary ===
    def visit_Dict(self, node: ast.Dict):  # noqa: N802
        pairs = []
        for key, value in zip(node.keys, node.values):
            if key is None:  # Handle dict unpacking
                continue
            key_js = self.visit(key)
            value_js = self.visit(value)
            pairs.append(f"{key_js}: {value_js}")
        return f"{{{', '.join(pairs)}}}"

    def visit_Tuple(self, node: ast.Tuple):  # noqa: N802
        elements = [self.visit(elt) for elt in node.elts]
        return f"[{', '.join(elements)}]"

    def get_expr_type(self, node: ast.AST) -> type | None:
        """Determine the type of an expression if possible."""
        if isinstance(node, ast.Constant):
            return type(node.value)
        elif isinstance(node, ast.Name):
            # First check if we have a stored type from type hints or assignments
            if node.id in self.var_types:
                return self.var_types[node.id]
            return None
        elif isinstance(node, ast.BinOp):
            # For binary operations, check if both operands have the same known type
            left_type = self.get_expr_type(node.left)
            right_type = self.get_expr_type(node.right)
            if left_type == right_type and left_type is not None:
                return left_type
            return None
        elif isinstance(node, ast.Call):
            # We can't determine the return type of function calls yet
            return None
        elif isinstance(node, ast.List):
            return list
        elif isinstance(node, ast.Dict):
            return dict
        return None


def transpile(fn: Callable) -> str:
    """
    Transpiles a Python function to JavaScript and returns the JavaScript code as a string.
    """
    try:
        source = inspect.getsource(fn)
        source = textwrap.dedent(source)
    except Exception as e:
        raise TranspilerError(
            message="Could not retrieve source code from the function."
        ) from e

    # Parse the source code into an AST.
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise TranspilerError(message="Could not parse function source.") from e

    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            func_node = node
            break

    if func_node is None:
        raise TranspilerError(
            message="No function or lambda definition found in the provided source."
        )

    visitor = PythonToJSVisitor()
    visitor.source_lines = source.splitlines()

    if isinstance(func_node, ast.Lambda):
        # Create a simple function wrapper for the lambda
        args = [arg.arg for arg in func_node.args.args]
        visitor.js_lines.append(f"function ({', '.join(args)}) " + "{")
        visitor.indent_level += 1
        visitor.js_lines.append(
            f"{visitor.indent()}return {visitor.visit(func_node.body)};"
        )
        visitor.indent_level -= 1
        visitor.js_lines.append("}")
    else:
        visitor.visit(func_node)

    if visitor.issues:
        raise TranspilerError(issues=visitor.issues)

    return "\n".join(visitor.js_lines)


# === Example Usage ===
def example_function(x, y):
    z = x + y
    return z


if __name__ == "__main__":
    js_code = transpile(example_function)
    print(js_code)
