from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_notebook_linear.py NOTEBOOK.ipynb")
    notebook_path = Path(sys.argv[1]).resolve()
    try:
        from IPython.display import display
    except Exception:
        def display(*objects: object) -> None:
            for obj in objects:
                print(obj)

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "display": display,
    }
    nb = nbformat.read(notebook_path, as_version=4)
    for index, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue
        print(f"[{notebook_path.name}] executing code cell {index}")
        exec(compile(cell.source, f"{notebook_path}#cell{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
