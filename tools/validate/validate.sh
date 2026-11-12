#!/usr/bin/env bash
# Runs inside the lme4-validate image with the repository mounted at /work.
#   validate              # R references + notebooks + tests
#   validate refs         # only regenerate the R reference results
#   validate notebooks    # only execute the example notebooks
#   validate tests        # only the pytest suite (against the installed package)
#   validate notebook 01_x.ipynb [...]   # selected notebooks
# LME4_REINSTALL=1 reinstalls the package from the mounted repository first.
set -euo pipefail
what="${1:-all}"
cd /work
if [[ "${LME4_REINSTALL:-0}" == 1 ]]; then
  echo "== reinstalling lme4 from a private copy of /work"
  # build from a private copy: several containers may validate the same mounted
  # repository at the same time and must not share /work/build
  rm -rf /tmp/src && mkdir -p /tmp/src
  tar -C /work --exclude=./build --exclude=./.venv --exclude=./.git       --exclude='*.egg-info' --exclude='__pycache__' -cf - . | tar -C /tmp/src -xf -
  pip install -q --force-reinstall --no-deps /tmp/src
fi
if [[ "$what" == notebook ]]; then
  shift; python examples/run_notebooks.py "$@"; exit $?
fi
if [[ "$what" == all || "$what" == refs ]]; then
  echo "== R reference results (R $(Rscript -e 'cat(as.character(getRversion()))'), lme4 $(Rscript -e 'cat(as.character(packageVersion("lme4")))'))"
  (cd tests/reference && Rscript make_reference.R specs.json reference.json >/dev/null && Rscript make_extras.R >/dev/null)
  for f in examples/r/*.R; do
    [ -e "$f" ] || continue
    echo "   $f"; (cd examples && Rscript "r/$(basename "$f")" >/dev/null)
  done
fi
if [[ "$what" == all || "$what" == notebooks ]]; then
  echo "== notebooks"; python examples/run_notebooks.py
fi
if [[ "$what" == all || "$what" == tests ]]; then
  echo "== tests (installed package, not src/)"
  python -m pytest -q -p no:cacheprovider -W ignore -o pythonpath= tests
fi
