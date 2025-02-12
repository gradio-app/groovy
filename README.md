<p align="center">
    <a href="https://pypi.org/project/grompy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/grompy"></a>
    <img alt="Python version" src="https://img.shields.io/badge/python-3.10+-important">
    <a href="https://github.com/abidlabs/grompy/actions/workflows/format.yml"><img alt="Format" src="https://github.com/abidlabs/grompy/actions/workflows/format.yml/badge.svg"></a>
    <a href="https://github.com/abidlabs/grompy/actions/workflows/test.yml"><img alt="Test" src="https://github.com/abidlabs/grompy/actions/workflows/test.yml/badge.svg"></a>
</p>


<h1 align="center"> 🐻 Grompy</h1>


Grompy is a Python-to-JavaScript transpiler, meaning that it converts Python functions to their JavaScript equivalents. It is used in the [Gradio](https://gradio.app) library, so that developers can write functions in Python and have them run as fast as client-side JavaScript ⚡. Instead of aiming for full coverage of Python features, Grompy prioritizes **clear error reporting** for unsupported Python code, making it easier for developers to modify their functions accordingly.

### 🚀 Features
- Converts simple Python functions into JavaScript equivalents.
- Supports a subset of the Python standard library along with some Gradio-specific classes.
- Provides **complete error reporting** when a function can't be transpiled (either due to no equivalent in JavaScript or ambiguity).

### 📦 Installation
Install Grompy via pip:
```bash
pip install grompy
```

### 🔧 Usage
```python
from grompy import transpile

def sum_range(n: int):
    total = 0
    for i in range(n):
        total = total + i
    return total

js_code = transpile(sum)
print(js_code)
```
produces:

```
function sum_range(n) {
    let total = 0;
    for (let i of Array.from({length: n}, (_, i) => i)) {
        total = (total + i);
    }
    return total;
```

Note that the JavaScript function is not necessarily minimized or optimized (yet) but it should return exactly the same value when called with the same arguments. While the internal implementation of the transpiled JavaScript function may differ slightly from the Python version (e.g. Python tuples may get converted to JS arrays), Grompy tries to guarantee that the return values will be identical for the same inputs.

If Grompy encounters unsupported syntax, it will **complain clearly** (throw a `TranspilationError` with all of the issues along with line numbers and the code that caused the issue, making it easy for developers to fix their code.


```python
def example_function(x, y):
    z = x + y
    if z > 10:
        print(z)
    else:
        print(0)

transpile(example_function)
```

raises:

```
TranspilerError: 2 issues found:

* Line 4: Unsupported function "print()"
>> print(z)
* Line 6: Unsupported function "print()"
>> print(0)
```

### 🤔 Ambiguity
Grompy takes a conservative approach when encountering ambiguous types. Instead of making assumptions about types, it will raise a `TranspilationError`. For example, a simple sum function without type hints will fail:

```python
def sum(a, b):
    return a + b

transpile(sum)  # Raises TranspilationError: Cannot determine types for parameters 'a' and 'b'
```

This is because `+` could mean different things in JavaScript depending on the types (string concatenation vs numeric addition vs. custom behavior for a custom class). Adding type hints (which is **strongly recommended** for all usage) resolves the ambiguity:

```python
def sum(a: int, b: int):
    return a + b

transpile(sum)  # Works! Produces: function sum(a, b) { return (a + b); }
```

### 🔥 Supported Syntax

**General**

| Python Syntax | JavaScript Equivalent |
|--------------|----------------------|
| `def function(x: int)` | `function function(x)` |
| `x + y`, `x - y`, `x * y`, `x / y` | `(x + y)`, `(x - y)`, `(x * y)`, `(x / y)` |
| `x > y`, `x < y`, `x >= y`, `x <= y` | `(x > y)`, `(x < y)`, `(x >= y)`, `(x <= y)` |
| `x == y`, `x != y` | `(x === y)`, `(x !== y)` |
| `x in list`, `x not in list` | `list.includes(x)`, `!list.includes(x)` |
| `x = value` | `let x = value` |
| `if condition: ... elif: ... else: ...` | `if (condition) { ... } else if { ... } else { ... }` |
| `for x in range(n)` | `for (let x of Array.from({length: n}, (_, i) => i))` |
| `for x in list` | `for (let x of list)` |
| `while condition` | `while (condition)` |
| `[x for x in list if condition]` | `list.filter(x => condition).map(x => x)` |
| `len(list)` | `list.length` |
| `len(dict)` | `Object.keys(dict).length` |
| `[1, 2, 3]` | `[1, 2, 3]` |
| `(1, 2, 3)` | `[1, 2, 3]` |
| `{"key": "value"}` | `{"key": "value"}` |
| `x and y`, `x or y` | `(x && y)`, `(x || y)` |
| `None` | `null` |


**Gradio-specific syntax**

// ... existing code ...
**Gradio-specific syntax**

| Python Syntax | JavaScript Equivalent |
|--------------|----------------------|
| `gr.Component(param=value)` | `{"param": "value", "__type__": "update"}` |


### 📜 License
Grompy is open-source under the [MIT License](https://github.com/abidlabs/grompy/blob/main/LICENSE).

---
Contributions to increase coverage of the Python library that Grompy can transpile are welcome! We welcome AI-generated PRs if the rationale is clear to follow, PRs are not too large in scope, and tests are included.
