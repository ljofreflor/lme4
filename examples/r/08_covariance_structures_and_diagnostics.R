## R reference for examples/08_covariance_structures_and_diagnostics.ipynb
## Run (from examples/):  Rscript r/08_covariance_structures_and_diagnostics.R
suppressPackageStartupMessages({library(lme4); library(jsonlite)})
out <- list()
un <- function(x) unname(as.vector(x))
txt <- function(x) capture.output(print(x))
## fit, recording warnings and messages instead of printing them
quiet <- function(expr) {
    w <- character(0); m <- character(0)
    val <- withCallingHandlers(expr,
        warning = function(e) { w <<- c(w, conditionMessage(e)); invokeRestart("muffleWarning") },
        message = function(e) { m <<- c(m, conditionMessage(e)); invokeRestart("muffleMessage") })
    list(value = val, warnings = w, messages = m)
}
mstats <- function(m) list(logLik = as.numeric(logLik(m)), df = attr(logLik(m), "df"),
                           AIC = AIC(m), fixef = un(fixef(m)), par = un(getME(m, "par")),
                           par_names = names(getME(m, "par")), theta = un(getME(m, "theta")),
                           sigma = sigma(m), singular = isSingular(m),
                           VarCorr = txt(VarCorr(m)))

## ================================================================ (a) structured covariances
## test-covariance_structures.R: same model with us / cs / diag (ML)
fm1      <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy, REML = FALSE)
fm1.us   <- lmer(Reaction ~ Days + us(Days | Subject), sleepstudy, REML = FALSE)
fm1.cs   <- lmer(Reaction ~ Days + cs(Days | Subject), sleepstudy, REML = FALSE)
fm1.diag <- lmer(Reaction ~ Days + diag(Days | Subject), sleepstudy, REML = FALSE)
out$fm1 <- mstats(fm1); out$fm1_us <- mstats(fm1.us)
out$fm1_cs <- mstats(fm1.cs); out$fm1_diag <- mstats(fm1.diag)
out$fm1_us_VarCorr <- un(as.matrix(VarCorr(fm1.us)$Subject))
out$fm1_cs_VarCorr <- un(as.matrix(VarCorr(fm1.cs)$Subject))
out$fm1_diag_VarCorr <- un(as.matrix(VarCorr(fm1.diag)$Subject))
out$aic_table <- txt(anova(fm1.diag, fm1.cs, fm1.us))

## vignette("covariance_structures"): REML, cs(1 + Days | Subject)
fm1.csR <- lmer(Reaction ~ Days + cs(1 + Days | Subject), sleepstudy)
out$vig_cs <- mstats(fm1.csR)
out$vig_cs_getReCovs <- txt(getReCovs(fm1.csR))
out$vig_cs_VarCorr_Subject <- un(as.matrix(VarCorr(fm1.csR)$Subject))
out$vig_cs_summary <- txt(summary(fm1.csR))

## one random effect per day: structured 10 x 10 covariance matrices
ss <- sleepstudy
m_cs_hom <- lmer(Reaction ~ Days + cs(0 + factor(Days) | Subject, hom = TRUE), ss, REML = FALSE)
m_cs     <- lmer(Reaction ~ Days + cs(0 + factor(Days) | Subject), ss, REML = FALSE)
m_ar1    <- lmer(Reaction ~ Days + ar1(0 + factor(Days) | Subject), ss, REML = FALSE)
m_slope  <- lmer(Reaction ~ Days + (Days | Subject), ss, REML = FALSE)
out$m_cs_hom <- mstats(m_cs_hom); out$m_cs <- mstats(m_cs); out$m_ar1 <- mstats(m_ar1)
out$m_ar1_VarCorr_Subject <- un(as.matrix(VarCorr(m_ar1)$Subject))
out$m_ar1_summary <- txt(summary(m_ar1))
out$aic_days <- un(AIC(m_cs_hom, m_cs, m_ar1, m_slope)$AIC)
out$df_days <- un(AIC(m_cs_hom, m_cs, m_ar1, m_slope)$df)
out$m_ar1_rePCA_sdev <- un(rePCA(m_ar1)$Subject$sdev)

## ================================================================ (b) diagnostics
## --- singular fits
fd2 <- quiet(lmer(Yield ~ 1 + (1 | Batch), Dyestuff2))
out$dy2_messages <- fd2$messages
out$dy2 <- mstats(fd2$value)

fcake <- quiet(lmer(angle ~ recipe + temp + (recipe | replicate), cake))
mc <- fcake$value
out$cake_messages <- fcake$messages
out$cake <- mstats(mc)
out$cake_VarCorr_replicate <- un(as.matrix(VarCorr(mc)$replicate))
pc <- rePCA(mc)
out$cake_rePCA_sdev <- un(pc$replicate$sdev)
out$cake_rePCA_rotation <- un(pc$replicate$rotation)
out$cake_rePCA_importance <- un(summary(pc)$replicate$importance)
out$cake_rePCA_print <- txt(pc)
mc_diag <- quiet(lmer(angle ~ recipe + temp + diag(recipe | replicate), cake))$value
mc_cs <- quiet(lmer(angle ~ recipe + temp + cs(recipe | replicate, hom = TRUE), cake))$value
mc_int <- quiet(lmer(angle ~ recipe + temp + (1 | replicate), cake))$value
out$cake_diag <- mstats(mc_diag); out$cake_cs <- mstats(mc_cs); out$cake_int <- mstats(mc_int)
out$cake_anova <- txt(anova(mc_int, mc_cs, mc_diag, mc, refit = FALSE))

fm1R <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
pf <- rePCA(fm1R)
out$fm1R_rePCA_sdev <- un(pf$Subject$sdev)
out$fm1R_rePCA_rotation <- un(pf$Subject$rotation)
out$fm1R_singular <- isSingular(fm1R)

## --- convergence (?convergence)
out$fm1R_theta <- un(getME(fm1R, "theta"))
out$fm1R_logLik <- as.numeric(logLik(fm1R))
out$fm1R_gradient <- un(fm1R@optinfo$derivs$gradient)
out$fm1R_Hessian <- un(fm1R@optinfo$derivs$Hessian)
out$fm1R_rcond <- Matrix::rcond(fm1R@optinfo$derivs$Hessian)
strict_tol <- lmerControl(optCtrl = list(xtol_abs = 1e-8, ftol_abs = 1e-8))
fm1.tol <- update(fm1R, control = strict_tol)
out$tol_theta <- un(getME(fm1.tol, "theta")); out$tol_logLik <- as.numeric(logLik(fm1.tol))
ss.CS <- transform(sleepstudy, Days = as.vector(scale(Days)))
fm1.CS <- update(fm1R, data = ss.CS)
out$CS_fixef <- un(fixef(fm1.CS)); out$CS_logLik <- as.numeric(logLik(fm1.CS))
pars <- getME(fm1R, "theta")
fm1.restart <- update(fm1R, start = pars)
out$restart_theta <- un(getME(fm1.restart, "theta"))
## Richardson-free check: the deviance function at the optimum
devfun <- update(fm1R, devFunOnly = TRUE)
out$devfun_at_theta <- devfun(pars)

## a model with a (spurious) convergence warning: quadratic random slopes
fq <- quiet(lmer(Reaction ~ Days + I(Days^2) + (Days + I(Days^2) | Subject), sleepstudy))
out$quad_warnings <- fq$warnings
out$quad <- mstats(fq$value)
out$quad_gradient <- un(fq$value@optinfo$derivs$gradient)

## --- rescaling predictors: vignette("autoscale")
set.seed(1)
sleepstudy$var1 <- runif(nrow(sleepstudy), 1e6, 1e7)
out$var1 <- sleepstudy$var1
f1 <- quiet(lmer(Reaction ~ var1 + Days + (Days | Subject), sleepstudy))
fit1 <- f1$value
out$scale_warnings <- f1$warnings
fit2 <- lmer(Reaction ~ var1 + Days + (Days | Subject),
             control = lmerControl(autoscale = TRUE), sleepstudy)
out$fit1_fixef <- un(fixef(fit1)); out$fit2_fixef <- un(fixef(fit2))
out$fit1_vcov <- un(as.matrix(vcov(fit1))); out$fit2_vcov <- un(as.matrix(vcov(fit2)))
out$fit1_logLik <- as.numeric(logLik(fit1)); out$fit2_logLik <- as.numeric(logLik(fit2))

## --- allFit (?allFit), restricted to optimizers needing no extra packages
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
mt <- data.frame(optimizer = c("bobyqa", "Nelder_Mead", "nlminbwrap", "nloptwrap", "nloptwrap"),
                 method = c("", "", "", "NLOPT_LN_NELDERMEAD", "NLOPT_LN_BOBYQA"))
gm_all <- allFit(gm1, meth.tab = mt, verbose = FALSE)
sa <- summary(gm_all)
out$af_names <- names(sa$which.OK)
out$af_which_OK <- un(sa$which.OK)
out$af_llik <- un(sa$llik)
out$af_fixef <- un(sa$fixef)
out$af_theta <- un(sa$theta)
out$af_sdcor <- un(sa$sdcor)
out$gm1_logLik <- as.numeric(logLik(gm1))

## --- lmList (?lmList)
fm.plm <- lmList(Reaction ~ Days | Subject, sleepstudy)
out$plm_coef <- un(as.matrix(coef(fm.plm)))
out$plm_subjects <- rownames(coef(fm.plm))
out$plm_sigma <- sigma(fm.plm)
ci <- confint(fm.plm)
out$plm_ci <- un(ci)
fm.2 <- update(fm.plm, pool = FALSE)
out$plm2_coef <- un(as.matrix(coef(fm.2)))

## --- influence, Cook's distance, hat values (?influence.merMod, ?hatvalues.merMod)
inf_fm1 <- influence(fm1R, "Subject")
out$inf_cooks <- un(cooks.distance(inf_fm1))
out$inf_dfbeta <- un(dfbeta(inf_fm1))
out$inf_dfbetas <- un(dfbetas(inf_fm1))
out$inf_varcov <- un(dfbeta(inf_fm1, which = "var.cov"))
out$hat <- un(hatvalues(fm1R))
out$cooks_obs <- un(cooks.distance(fm1R))

## --- residuals
out$res_resp <- un(residuals(fm1R))
out$res_pearson_scaled <- un(residuals(fm1R, type = "pearson", scaled = TRUE))
out$fitted <- un(fitted(fm1R))
out$res_quantiles <- un(quantile(residuals(fm1R, scaled = TRUE)))
out$ranef_Subject <- un(as.matrix(ranef(fm1R)$Subject))

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/08_covariance_structures_and_diagnostics.json")
cat("wrote reference/08_covariance_structures_and_diagnostics.json\n")
