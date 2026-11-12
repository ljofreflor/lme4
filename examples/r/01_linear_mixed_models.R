## R reference results for examples/01_linear_mixed_models.ipynb
## Run from examples/:  Rscript r/01_linear_mixed_models.R
suppressPackageStartupMessages({library(lme4); library(jsonlite); library(lattice)})

## summary of a fitted LMM, as a list
fitinfo <- function(m) {
    vc <- as.data.frame(VarCorr(m))
    cf <- coef(summary(m))
    list(REML = isREML(m),
         REMLcrit = if (isREML(m)) REMLcrit(m) else NA,
         deviance = if (isREML(m)) NA else deviance(m),
         logLik = as.numeric(logLik(m)),
         df = attr(logLik(m), "df"),
         AIC = AIC(m), BIC = BIC(m),
         theta = unname(getME(m, "theta")),
         theta_names = names(getME(m, "theta")),
         sigma = sigma(m),
         fixef = as.list(fixef(m)),
         se = unname(cf[, "Std. Error"]),
         tvalue = unname(cf[, "t value"]),
         vcov = unname(as.matrix(vcov(m))),
         vc_grp = vc$grp, vc_var1 = vc$var1, vc_var2 = vc$var2,
         vc_vcov = vc$vcov, vc_sdcor = vc$sdcor,
         isSingular = isSingular(m),
         nobs = nobs(m),
         ngrps = as.list(ngrps(m)))
}
## random effects with conditional SDs, and coef()
reinfo <- function(m) {
    rr <- as.data.frame(ranef(m, condVar = TRUE))
    list(grpvar = as.character(rr$grpvar), term = as.character(rr$term),
         grp = as.character(rr$grp), condval = rr$condval, condsd = rr$condsd,
         coef = lapply(coef(m), function(d) c(list(levels = rownames(d)), as.list(d))))
}
anovainfo <- function(a) {
    list(models = rownames(a), npar = a$npar, AIC = a$AIC, BIC = a$BIC,
         logLik = a$logLik, deviance = a[["-2*log(L)"]],
         Chisq = a$Chisq, Df = a$Df, p = a[["Pr(>Chisq)"]])
}
out <- list()

## ---------------------------------------------------------------- sleepstudy, ?lmer
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
out$fm1 <- c(fitinfo(fm1), reinfo(fm1))
out$fm1$scaled_resid_quantiles <- unname(quantile(residuals(fm1, type = "pearson", scaled = TRUE)))
out$fm1$corr_fixef <- unname(cov2cor(as.matrix(vcov(fm1)))[1, 2])

fm1_ML <- update(fm1, REML = FALSE)
out$fm1_ML <- fitinfo(fm1_ML)

fm2 <- lmer(Reaction ~ Days + (Days || Subject), sleepstudy)
out$fm2 <- c(fitinfo(fm2), reinfo(fm2))
out$fm2$formula <- deparse(formula(fm2))
out$anova_fm1_fm2 <- anovainfo(anova(fm1, fm2))

## random intercept only, and the full comparison
fm0 <- lmer(Reaction ~ Days + (1 | Subject), sleepstudy)
out$fm0 <- c(fitinfo(fm0), reinfo(fm0))
out$anova_fm0_fm2_fm1 <- anovainfo(anova(fm0, fm2, fm1))
## anova without refitting (REML criteria compared directly)
out$anova_fm0_fm2_REML <- anovainfo(anova(fm0, fm2, refit = FALSE))

## ---------------------------------------------------------------- ?sleepstudy (subset)
fm1s <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy, subset = Days >= 2)
out$fm1s <- fitinfo(fm1s)
fm2s <- lmer(Reaction ~ Days + (1|Subject) + (0+Days|Subject), sleepstudy, subset = Days >= 2)
out$fm2s <- fitinfo(fm2s)

## ---------------------------------------------------------------- Dyestuff / Dyestuff2
d1 <- lmer(Yield ~ 1|Batch, Dyestuff)
out$dye1 <- c(fitinfo(d1), reinfo(d1))
d1ML <- update(d1, REML = FALSE)
out$dye1_ML <- fitinfo(d1ML)
d2 <- lmer(Yield ~ 1|Batch, Dyestuff2)
out$dye2 <- c(fitinfo(d2), reinfo(d2))
out$dye2$messages <- d2@optinfo$conv$lme4$messages
out$dye2_lm_sigma <- sigma(lm(Yield ~ 1, Dyestuff2))

## ---------------------------------------------------------------- plots: the plotted data
## plot(fm1): pearson residuals vs fitted values
p <- plot(fm1)
out$plot_fm1 <- list(x = as.numeric(p$panel.args[[1]]$x), y = as.numeric(p$panel.args[[1]]$y),
                     xlab = p$xlab, ylab = p$ylab)
## plot(fm1, form, id): identified points
idx <- abs(resid(fm1, type = "pearson"))/sigma(fm1) > -qnorm(0.05/2)
out$plot_fm1_id <- list(which = which(idx), labels = as.character(sleepstudy$Subject[idx]))
## plot by panels
p <- plot(fm1, resid(., scaled = TRUE) ~ fitted(.) | Subject, abline = 0)
out$plot_fm1_panels <- list(levels = p$condlevels[[1]],
                            n = sapply(p$panel.args, function(a) length(a$x)))
## box-plots of residuals by subject
p <- plot(fm1, Subject ~ resid(., scaled = TRUE))
out$plot_fm1_bw <- list(x = as.numeric(p$panel.args[[1]]$x), y = as.character(p$panel.args[[1]]$y))

## dotplot(ranef(fm1, condVar = TRUE))
rr <- ranef(fm1, condVar = TRUE)
dp <- dotplot(rr)$Subject
ss <- lme4:::asDf0(rr, "Subject")
out$dotplot_fm1 <- list(levels = levels(ss$.nn), panels = as.character(dp$condlevels[[1]]),
                        x = lapply(dp$panel.args, function(a) as.numeric(a$x)),
                        y = lapply(dp$panel.args, function(a) as.character(a$y)),
                        se = ss$se, xlim = dp$x.limits)
## qqmath(ranef(fm1, condVar = TRUE))
qp <- qqmath(rr)$Subject
out$qqmath_fm1 <- list(panels = as.character(qp$condlevels[[1]]),
                       x = lapply(qp$panel.args, function(a) as.numeric(a$x)),
                       y = lapply(qp$panel.args, function(a) as.numeric(a$y)))
## qqmath(fm1): residual QQ plot
qr <- qqmath(fm1)
out$qqmath_resid <- list(x = as.numeric(qr$panel.args[[1]]$x))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/01_linear_mixed_models.json")
