from pathlib import Path
import ast


def test_deepface_analyze_is_never_referenced_in_sidecar() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "analyze":
            raise AssertionError("DeepFace.analyze() is strictly prohibited by BR-028.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and ".analyze(" in node.value:
            raise AssertionError("analyze call signature is prohibited in sidecar.")
