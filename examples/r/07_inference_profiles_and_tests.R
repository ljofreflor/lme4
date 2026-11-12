## R reference results for examples/07_inference_profiles_and_tests.ipynb
## Run from examples/:  Rscript r/07_inference_profiles_and_tests.R
suppressPackageStartupMessages({library(lme4); library(jsonlite); library(splines)})

out <- list(R_version = paste(R.version$major, R.version$minor, sep = "."),
            lme4_version = as.character(packageVersion("lme4")))

anova_tab <- function(a) list(rownames = rownames(a), npar = a$npar, AIC = a$AIC, BIC = a$BIC,
                              logLik = a$logLik, deviance = a[["-2*log(L)"]],
                              Chisq = a$Chisq, Df = a$Df, p = a[["Pr(>Chisq)"]])
ci_tab <- function(ci) list(rownames = rownames(ci), lower = unname(ci[, 1]), upper = unname(ci[, 2]))

## ------------------------------------------------------------- sleepstudy LRTs
fm1  <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
fm2  <- lmer(Reaction ~ Days + (Days || Subject), sleepstudy)
fm0  <- lmer(Reaction ~ 1 + (Days | Subject), sleepstudy)
fm1ML <- refitML(fm1)

out$anova_fixed  <- anova_tab(anova(fm0, fm1))
out$anova_corr   <- anova_tab(anova(fm2, fm1))
out$anova_corr_REML <- anova_tab(anova(fm2, fm1, refit = FALSE))
d1 <- drop1(fm1, test = "Chisq")
out$drop1 <- list(rownames = rownames(d1), npar = d1$npar, AIC = d1$AIC,
                  LRT = d1$LRT, p = d1[["Pr(Chi)"]])

## drop1(test = "user"): a summary function returning the LRT from anova()
LRTSumFun <- function(object, objectDrop, ...) {
    nms <- c("Chisq", "Df", "p.value")
    if (missing(objectDrop)) return(setNames(rep(NA_real_, 3), nms))
    a <- anova(objectDrop, object)
    setNames(c(a$Chisq[2], a$Df[2], a[["Pr(>Chisq)"]][2]), nms)
}
du <- drop1(fm1, test = "user", sumFun = LRTSumFun)
out$drop1_user <- list(rownames = rownames(du), Chisq = du$Chisq, Df = du$Df, p = du$p.value)

## parametric-bootstrap LRT for Days (simulate from the null model, refit both)
set.seed(101)
nsim_pb <- 200
fm0ML <- refitML(fm0)
lrt_obs <- as.numeric(2 * (logLik(fm1ML) - logLik(fm0ML)))
ysim <- simulate(fm0ML, nsim = nsim_pb)
lrt_sim <- sapply(ysim, function(y) {
    m0 <- refit(fm0ML, y); m1 <- refit(fm1ML, y)
    as.numeric(2 * (logLik(m1) - logLik(m0)))
})
out$pb_lrt <- list(obs = lrt_obs, nsim = nsim_pb,
                   p = (sum(lrt_sim >= lrt_obs) + 1) / (nsim_pb + 1),
                   sim_mean = mean(lrt_sim), sim_q95 = unname(quantile(lrt_sim, 0.95)))

## REML vs ML
vcinfo <- function(m) { vc <- as.data.frame(VarCorr(m))
    list(sdcor = vc$sdcor, fixef = unname(fixef(m)),
         se = unname(sqrt(diag(as.matrix(vcov(m))))),
         logLik = as.numeric(logLik(m)), deviance = deviance(m, REML = FALSE),
         REMLcrit = if (isREML(m)) REMLcrit(m) else NA) }
out$fm1_REML <- vcinfo(fm1)
out$fm1_ML   <- vcinfo(fm1ML)

## Wald intervals (computed before profiling)
out$fm1_wald <- ci_tab(confint(fm1, method = "Wald"))
## t statistics with normal-approximation p-values
cf <- coef(summary(fm1))
out$fm1_coef <- list(t = unname(cf[, "t value"]))

## ------------------------------------------------------------- Dyestuff profile
fm01ML <- lmer(Yield ~ 1 | Batch, Dyestuff, REML = FALSE)
tprb <- profile(fm01ML, optimizer = "Nelder_Mead", which = "beta_")
out$dyestuff_beta_ci <- ci_tab(confint(tprb))
tpr <- profile(fm01ML, signames = FALSE)
out$dyestuff_ci <- ci_tab(confint(tpr))
out$dyestuff_ci_90 <- ci_tab(confint(tpr, level = 0.90))

nms <- names(attr(tpr, "forward"))
lev <- sqrt(qchisq(c(50, 80, 90, 95, 99)/100, 1))
zeta <- c(-rev(lev), 0, lev)
out$dyestuff_xyplot <- list(zeta = zeta,
    pval = lapply(setNames(nms, nms), function(n) lme4:::predy(attr(tpr, "backward")[[n]], zeta)))

dd <- lme4:::dens(tpr)
out$dyestuff_dens <- list(pnm = as.character(dd$pnm), pval = dd$pval, density = dd$density)
## with signames = FALSE the names are "sd_(Intercept)|Batch" and "sigma", which
## logProf()/varianceProf() (matching "^\.sig") leave unchanged: use the default names
tprS <- profile(fm01ML)
out$dyestuff_ciS <- ci_tab(confint(tprS))
out$dyestuff_logProf_unchanged <- identical(names(logProf(tpr, ranef = FALSE)), names(tpr))
vp <- varianceProf(tprS)
out$dyestuff_var_ci <- ci_tab(confint(vp))
ddv <- lme4:::dens(vp)
out$dyestuff_var_dens <- list(pnm = as.character(ddv$pnm), pval = ddv$pval, density = ddv$density)
lp <- logProf(tprS, ranef = FALSE)
out$dyestuff_log_ci <- ci_tab(confint(lp))
lp2 <- logProf(tprS)
out$dyestuff_log2_ci <- ci_tab(confint(lp2))

## splom.thpr internals: parameter ranges and the contours of pair (2, 1)
levs2 <- sqrt(qchisq(c(50, 80, 90, 95, 99)/100, 2))
mlev <- max(levs2)
spl <- attr(tpr, "forward"); bsp <- attr(tpr, "backward")
frange <- sapply(spl, function(x) range(x$knots))
pfr <- do.call(cbind, lapply(bsp, lme4:::predy, c(-mlev, mlev)))
pfr[1, ] <- pmax.int(pfr[1, ], frange[1, ], na.rm = TRUE)
pfr[2, ] <- pmin.int(pfr[2, ], frange[2, ], na.rm = TRUE)
x <- tpr[, c(nms, ".par")]
fr <- x
for (nm in nms) fr[[nm]] <- lme4:::predy(spl[[nm]], fr[[nm]])
conts <- list()
for (pair in list(c(1, 2), c(1, 3), c(2, 3))) {
    i <- pair[1]; j <- pair[2]
    fri <- subset(fr, .par == nms[i]); frj <- subset(fr, .par == nms[j])
    sij <- interpSpline(fri[, i], fri[, j]); sji <- interpSpline(frj[, j], frj[, i])
    ll <- lme4:::cont(sij, sji, levs2)
    conts[[paste(j, i, sep = "_")]] <- list(tki = ll$tki$y, tkj = ll$tkj$y,
        pts1 = as.vector(ll$pts[, , 1]), pts2 = as.vector(ll$pts[, , 2]))
}
out$dyestuff_splom <- list(pfr_lower = unname(pfr[1, ]), pfr_upper = unname(pfr[2, ]),
                           conts = conts)

## REML fit of the same model
fm01 <- lmer(Yield ~ 1 | Batch, Dyestuff)
out$dyestuff_REML_ci <- ci_tab(confint(fm01))

## ------------------------------------------------------------- sleepstudy intervals
pr1 <- profile(fm1)
out$fm1_profile_ci <- ci_tab(confint(pr1))
fm1b <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
set.seed(123)
bci <- suppressMessages(suppressWarnings(confint(fm1b, method = "boot", nsim = 500)))
out$fm1_boot_ci <- ci_tab(bci)
fm2b <- lmer(Reaction ~ Days + (Days || Subject), sleepstudy)
out$fm2_ci_maxpts8 <- ci_tab(confint(fm2b, maxpts = 8))

## ------------------------------------------------------------- GLMM (cbpp)
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), cbpp, family = binomial)
gm0 <- glmer(cbind(incidence, size - incidence) ~ 1 + (1 | herd), cbpp, family = binomial)
cg <- coef(summary(gm1))
out$gm1_coef <- list(z = unname(cg[, "z value"]), p = unname(cg[, "Pr(>|z|)"]))
out$gm1_anova <- anova_tab(anova(gm0, gm1))
dg <- drop1(gm1, test = "Chisq")
out$gm1_drop1 <- list(rownames = rownames(dg), npar = dg$npar, AIC = dg$AIC,
                      LRT = dg$LRT, p = dg[["Pr(Chi)"]])
out$gm1_wald_ci <- ci_tab(confint(gm1, method = "Wald"))
out$gm1_profile_ci <- ci_tab(confint(gm1))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/07_inference_profiles_and_tests.json")
cat("done\n")
