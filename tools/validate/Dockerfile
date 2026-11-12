# Validation environment: R 4.6.2 + lme4 2.1.0 (reference) and Python + lme4 for Python.
#   docker build -t lme4-validate -f tools/validate/Dockerfile .
# The Python package is installed from the repository source (default) or from PyPI:
#   docker build --build-arg LME4_FROM=pypi -t lme4-validate -f tools/validate/Dockerfile .
FROM rocker/r-ver:4.6.2
RUN Rscript -e 'install.packages(c("lme4", "jsonlite"), \
      repos = "https://packagemanager.posit.co/cran/__linux__/noble/2026-11-10")' \
 && Rscript -e 'stopifnot(packageVersion("lme4") == "2.1.0")'
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev g++ \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py && /opt/py/bin/pip install -q --upgrade pip \
 && /opt/py/bin/pip install -q nbformat nbclient ipykernel matplotlib pytest
ENV PATH=/opt/py/bin:$PATH
ARG LME4_FROM=source
COPY . /src
RUN if [ "$LME4_FROM" = "pypi" ]; then pip install -q --pre lme4; \
    else pip install -q /src; fi \
 && python -c "import lme4; print('lme4 for Python', lme4.__version__)"
WORKDIR /work
COPY tools/validate/validate.sh /usr/local/bin/validate
# prefer the (possibly newer) script of the mounted repository
ENTRYPOINT ["bash", "-c", "s=/work/tools/validate/validate.sh; [ -f $s ] || s=/usr/local/bin/validate; exec bash $s \"$@\"", "validate"]
