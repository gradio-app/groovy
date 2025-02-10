from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable


class TranspilerError(Exception):
    """Exception raised when transpilation fails or encounters ambiguous syntax."""

    def __init__(self, issues: list[tuple[int, str, str]] | None = None, message: str | None = None):
        self.issues = issues or []
        if message:
            super().__init__(message)
        else:
            issue_count = len(self.issues)
            issues_text = f"{issue_count} issue{'s' if issue_count != 1 else ''} found:\n\n"
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

    def add_issue(self, node: ast.AST, message: str) -> None:
        """Add a transpilation issue with source code context."""
        if hasattr(node, 'lineno'):
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
        # Extract parameter names. (Later we could add type hint checks.)
        params = [arg.arg for arg in node.args.args]
        header = f"function {node.name}({', '.join(params)}) " + "{"
        self.js_lines.append(header)
        self.indent_level += 1

        # Process the function body.
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
        # For standalone expressions, output them followed by a semicolon.
        expr = self.visit(node.value)
        self.js_lines.append(f"{self.indent()}{expr};")

    # === Assignment ===
    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        if len(node.targets) != 1:
            raise TranspilerError("Multiple assignment targets are not supported yet.")
        target = self.visit(node.targets[0])
        value = self.visit(node.value)

        # Only use 'let' for new variable declarations (Name nodes)
        # Skip 'let' for subscript assignments and reassignments
        if (
            isinstance(node.targets[0], ast.Name)
            and node.targets[0].id not in self.declared_vars
        ):
            self.declared_vars.add(node.targets[0].id)
            self.js_lines.append(f"{self.indent()}let {target} = {value};")
        else:
            self.js_lines.append(f"{self.indent()}{target} = {value};")

    # === Binary Operations ===
    def visit_BinOp(self, node: ast.BinOp):  # noqa: N802
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.visit(node.op)
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

        # For now, we only support single comparisons
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
            if node.func.id == 'range':
                args = [self.visit(arg) for arg in node.args]
                return self.visit_range(args)
            # All other direct function calls are not supported
            self.add_issue(node, f"Unsupported function \"{node.func.id}()\"")
            return ""
        
        # For method calls (like obj.method())
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return f"{func}({', '.join(args)})"

    # === Variable Name ===
    def visit_Name(self, node: ast.Name):  # noqa: N802
        return node.id

    # === Constants ===
    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        # Use repr() to generate a JS-friendly literal.
        return repr(node.value)

    # === For Loop ===
    def visit_For(self, node: ast.For):  # noqa: N802
        target = self.visit(node.target)
        iter_expr = self.visit(node.iter)

        # Generic for-of loop for all iterables
        self.js_lines.append(
            f"{self.indent()}for (let {target} of {iter_expr}) " + "{"
        )

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


def transpile(fn: Callable) -> str:
    """
    Transpiles a Python function to JavaScript and returns the JavaScript code as a string.
    """
    try:
        source = inspect.getsource(fn)
        source = textwrap.dedent(source)
    except Exception as e:
        raise TranspilerError(message="Could not retrieve source code from the function.") from e

    # Parse the source code into an AST.
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise TranspilerError(message="Could not parse function source.") from e

    # Find the first function definition in the AST.
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_node = node
            break

    if func_node is None:
        raise TranspilerError(message="No function definition found in the provided source.")

    # Visit the function node to generate JavaScript code.
    visitor = PythonToJSVisitor()
    visitor.source_lines = source.splitlines()
    visitor.visit(func_node)

    # If there were any issues, raise them all at once
    if visitor.issues:
        raise TranspilerError(issues=visitor.issues)

    return "\n".join(visitor.js_lines)


# === Example Usage ===
def example_function(x, y):
    z = x + y
    if z > 10:
        return z
    else:
        return 0


if __name__ == "__main__":
    try:
        js_code = transpile(example_function)
        print("Generated JavaScript Code:")
        print(js_code)
    except TranspilerError as err:
        print("Transpilation failed:", err)
