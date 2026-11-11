## Generates reference results with R's lme4 for the Python port's tests.
## Usage (in the lme4ref docker image):  Rscript make_reference.R specs.json reference.json
suppressMessages(library(lme4)); library(jsonlite)
args <- commandArgs(TRUE)
specs <- fromJSON(args[1], simplifyVector = FALSE)
out <- list()
for (s in specs) {
  data(list = s$data, package = "lme4"); dat <- get(s$data)
  if (!is.null(s$weights)) dat$`.w` <- with(dat, eval(parse(text = s$weights)))
  sub <- if (!is.null(s$subset)) with(dat, eval(parse(text = s$subset))) else rep(TRUE, nrow(dat))
  dat <- dat[sub, , drop = FALSE]
  f <- as.formula(s$formula)
  res <- tryCatch({
    m <- if (s$fun == "lmer") {
      if (is.null(s$weights)) lmer(f, dat, REML = if (is.null(s$REML)) TRUE else s$REML)
      else lmer(f, dat, REML = if (is.null(s$REML)) TRUE else s$REML, weights = .w)
    } else {
      fam <- get(s$family)(link = if (is.null(s$link)) get(s$family)()$link else s$link)
      glmer(f, dat, family = fam, nAGQ = if (is.null(s$nAGQ)) 1L else s$nAGQ)
    }
    vc <- as.data.frame(VarCorr(m))
    re <- ranef(m, condVar = TRUE)
    list(fixef = unname(fixef(m)), fixef_names = names(fixef(m)),
         theta = unname(getME(m, "theta")), theta_names = names(getME(m, "theta")),
         sigma = sigma(m), logLik = as.numeric(logLik(m)), df = attr(logLik(m), "df"),
         AIC = AIC(m), BIC = BIC(m),
         devcrit = if (isREML(m)) REMLcrit(m) else deviance(m),
         deviance_glmm = if (isGLMM(m)) deviance(m) else NULL,
         vcov = unname(as.matrix(vcov(m))), vcov_rx = unname(as.matrix(vcov(m, use.hessian = FALSE))),
         vc_vcov = vc$vcov, vc_sdcor = vc$sdcor,
         ranef = lapply(re, function(x) unname(as.matrix(x))),
         condsd = lapply(re, function(x) { pv <- attr(x, "postVar"); if (is.list(pv)) NULL else as.vector(apply(pv, 3, function(v) sqrt(diag(as.matrix(v))))) }),
         fitted = unname(fitted(m)), resid = unname(residuals(m)),
         singular = isSingular(m), nobs = nobs(m),
         summary = capture.output(print(summary(m))),
         print = capture.output(print(m)))
  }, error = function(e) list(error = conditionMessage(e)))
  out[[s$id]] <- res
  cat(s$id, if (!is.null(res$error)) paste("ERROR:", res$error) else "ok", "\n")
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null"), args[2])
