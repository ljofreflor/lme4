"""Execute all example notebooks (used by CI and locally):  python examples/run_notebooks.py"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

here = Path(__file__).parent
failed = []
selected = [here / a for a in sys.argv[1:]] or sorted(here.glob("[0-9][0-9]_*.ipynb"))
for nb_path in selected:
    nb = nbformat.read(nb_path, as_version=4)
    t = time.time()
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(here)}}).execute()
        print(f"ok    {nb_path.name}  ({time.time() - t:.0f} s)", flush=True)
    except Exception as e:  # noqa: BLE001
        failed.append(nb_path.name)
        print(f"FAIL  {nb_path.name}: {str(e).splitlines()[-1] if str(e) else e}", flush=True)
sys.exit(1 if failed else 0)
