<p align="center">
    <a href="https://pypi.org/project/grompy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/grompy"></a>
    <img alt="Python version" src="https://img.shields.io/badge/python-3.10+-important">
    <a href="https://github.com/abidlabs/grompy/actions/workflows/format.yml"><img alt="Format" src="https://github.com/abidlabs/grompy/actions/workflows/format.yml/badge.svg"></a>
    <a href="https://github.com/abidlabs/grompy/actions/workflows/test.yml"><img alt="Test" src="https://github.com/abidlabs/grompy/actions/workflows/test.yml/badge.svg"></a>
</p>


<h1 align="center"> 🐻 Grompy</h1>


Grompy is a Python-to-JavaScript transpiler meaning that it converts Python functions to their JavaScript equivlaents. It is designed for use in the [Gradio](https://gradio.app) library, so that developers can write functions in Python and have them run as fast as client-side JavaScript ⚡. Instead of aiming for full coverage of Python features, Grompy prioritizes **clear error reporting** for unsupported code, making it easier for developers to modify their functions accordingly.

### 🚀 Features
- Converts simple Python functions into JavaScript equivalents.
- Supports a subset of the Python standard library along with some Gradio-specific classes.
- Provides **detailed warnings** when something can't be transpiled.

### 📦 Installation
Install Grompy via pip:
```bash
pip install grompy
```

### 🔧 Usage
```python
from grompy import transpile

def sum(x, y):
    return x + y

js_code = transpile(sum)
print(js_code)

# function sum(x, y) {
#    return x + y;
# }
```

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

```
TranspilerError: 2 issues found:

* Line 4: Unsupported function "print()"
>> print(z)
* Line 6: Unsupported function "print()"
>> print(0)
```

### 📜 License
Grompy is open-source under the [MIT License](https://github.com/abidlabs/grompy/blob/main/LICENSE).

---
Contributions to increase coverage of the Python library that Grompy can transpile are welcome! We welcome AI-generated PRs if the rationale is clear to follow, PRs are not too large in scope, and tests are included.
