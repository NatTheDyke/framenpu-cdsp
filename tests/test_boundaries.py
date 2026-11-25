"""Import direction (docs/CONTRACTS.md in the framenpu repository).

Of framenpu, the backend imports only the plugin contracts and the error types, so that package
never sees the kernel's layout and this one never sees models or sessions. The backend's own
modules do not import framenpu_cdsp.research.
"""

import ast
import pathlib
import re
import sys
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src/framenpu_cdsp"
# imported third-party packages that only some modules need: the extra that declares each
EXTRAS = {"onnxruntime": "cpu"}
ALLOWED = {"framenpu.contracts", "framenpu.errors"}


def imports(path):
    """Absolute module names a file imports (any depth: lazy imports count)."""
    package = ("framenpu_cdsp",) + path.relative_to(PKG).parts[:-1]
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else ()
            module = ".".join(base + ((node.module,) if node.module else ()))
            yield from (
                (f"{module}.{a.name}" for a in node.names) if not node.module else (module,)
            )


def declared():
    """(required, by extra) distribution names of pyproject.toml."""

    def names(requirements):
        return {re.match(r"[\w.-]+", r)[0].lower() for r in requirements}

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project.get("optional-dependencies", {})
    return names(project.get("dependencies", [])), {k: names(v) for k, v in extras.items()}


class BoundaryTests(unittest.TestCase):
    def test_every_third_party_import_is_declared(self):
        required, extras = declared()
        for path in PKG.rglob("*.py"):
            found = {name.split(".")[0] for name in imports(path)}
            for name in found - set(sys.stdlib_module_names) - {"framenpu_cdsp"}:
                where = extras.get(EXTRAS[name], set()) if name in EXTRAS else required
                self.assertIn(name, where, f"{path.name} imports {name}, undeclared")

    def test_framenpu_is_reached_only_through_contracts_and_errors(self):
        for path in PKG.rglob("*.py"):
            for name in imports(path):
                if name.split(".")[0] == "framenpu":
                    self.assertIn(name, ALLOWED, path)

    def test_the_backend_does_not_import_research(self):
        for path in PKG.glob("*.py"):
            for name in imports(path):
                self.assertFalse(name.startswith("framenpu_cdsp.research"), path)

    def test_errors_are_framenpus_own_classes(self):
        import framenpu.errors
        import framenpu_cdsp.errors

        for name in framenpu_cdsp.errors.__all__:
            self.assertIs(getattr(framenpu_cdsp.errors, name), getattr(framenpu.errors, name))


if __name__ == "__main__":
    unittest.main()
