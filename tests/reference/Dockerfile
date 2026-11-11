# R environment used to produce the reference results (R 4.6.2 + lme4 2.1.0).
#   docker build -t lme4ref tests/reference
FROM rocker/r-ver:4.6.2
RUN Rscript -e 'install.packages(c("lme4", "jsonlite"), \
      repos = "https://packagemanager.posit.co/cran/__linux__/noble/2026-11-10")' \
 && Rscript -e 'stopifnot(packageVersion("lme4") == "2.1.0")'
WORKDIR /ref
