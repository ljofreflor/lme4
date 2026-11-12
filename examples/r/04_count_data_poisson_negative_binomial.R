## R reference for examples/04_count_data_poisson_negative_binomial.ipynb
## Run (from examples/):  Rscript r/04_count_data_poisson_negative_binomial.R
suppressPackageStartupMessages({library(lme4); library(jsonlite); library(MASS)})

ref <- list(R_version = R.version.string, lme4_version = as.character(packageVersion("lme4")))

fitinfo <- function(m) {
    out <- list(fixef = as.list(fixef(m)),
                theta = unname(getME(m, "theta")),
                logLik = as.numeric(logLik(m)),
                df = attr(logLik(m), "df"),
                deviance = deviance(m),
                AIC = AIC(m), BIC = BIC(m),
                sigma = sigma(m),
                df_residual = df.residual(m),
                sdcor = as.data.frame(VarCorr(m))$sdcor,
                se = unname(sqrt(diag(as.matrix(vcov(m))))),
                pearson_chisq = sum(residuals(m, type = "pearson")^2),
                messages = m@optinfo$conv$lme4$messages)
    if (is.null(out$messages)) out$messages <- character(0)
    out
}
anovainfo <- function(a) list(Chisq = a$Chisq[2], Df = a$Df[2], p = a[["Pr(>Chisq)"]][2],
                              AIC = a$AIC, npar = a$npar)

## ---- grouseticks (?grouseticks, Elston et al. 2001) --------------------------
data(grouseticks)
ref$agg_mean <- grouseticks_agg$meanTICKS
ref$agg_var  <- grouseticks_agg$varTICKS

## original formula of ?grouseticks (unscaled HEIGHT): deviance function at the
## parameter values stored in tests/elston.R, then the fit (convergence warnings)
form <- TICKS ~ YEAR + HEIGHT + (1|BROOD) + (1|INDEX) + (1|LOCATION)
dfun <- glmer(form, family = "poisson", data = grouseticks, devFunOnly = TRUE)
ref$elston_dev <- dfun(c(0.541509425632023, 0.750034415832756, 0.528723159081737,
                         11.3559080756861, 1.1804105508475, -0.978704335712111,
                         -0.0237607330254979))
full_mod1 <- glmer(form, family = "poisson", data = grouseticks)
ref$full_mod1 <- fitinfo(full_mod1)

## scaled HEIGHT, with and without the observation-level random effect (INDEX)
m1 <- glmer(TICKS ~ YEAR + scale(HEIGHT) + (1|BROOD) + (1|INDEX) + (1|LOCATION),
            family = poisson, data = grouseticks)
m0 <- glmer(TICKS ~ YEAR + scale(HEIGHT) + (1|BROOD) + (1|LOCATION),
            family = poisson, data = grouseticks)
ref$m1 <- fitinfo(m1)
ref$m0 <- fitinfo(m0)
ref$m1$ranef_BROOD <- ranef(m1)$BROOD[, 1]

## negative binomial instead of the OLRE
nb0 <- glmer.nb(TICKS ~ YEAR + scale(HEIGHT) + (1|BROOD) + (1|LOCATION), data = grouseticks)
ref$nb0 <- fitinfo(nb0)
ref$nb0$nb_theta <- getME(nb0, "glmer.nb.theta")
ref$nb0$family <- family(nb0)$family

ref$anova_m0_m1 <- anovainfo(anova(m0, m1))
ref$anova_m0_nb0 <- anovainfo(anova(m0, nb0))

## ---- ?glmer.nb example (simulated data saved for Python) -------------------
set.seed(101)
dd <- expand.grid(f1 = factor(1:3),
                  f2 = LETTERS[1:2], g=1:9, rep=1:15,
                  KEEP.OUT.ATTRS=FALSE)
mu <- 5*(-4 + with(dd, as.integer(f1) + 4*as.numeric(f2)))
dd$y <- rnbinom(nrow(dd), mu = mu, size = 0.5)
write.csv(dd, "data/04_glmer_nb_dd.csv", row.names = FALSE)
m.glm <- glm.nb(y ~ f1*f2, data = dd)
m.nb <- glmer.nb(y ~ f1*f2 + (1|g), data = dd)
ref$m_nb <- fitinfo(m.nb)
ref$m_nb$nb_theta <- getME(m.nb, "glmer.nb.theta")
ref$m_glm <- list(coef = unname(coef(m.glm)), theta = m.glm$theta,
                  logLik = as.numeric(logLik(m.glm)), df = attr(logLik(m.glm), "df"))

## ---- Poisson with an offset: ticks per brood, exposure = number of chicks ---
agg <- do.call(rbind, lapply(split(grouseticks, grouseticks$BROOD, drop = TRUE), function(d)
    data.frame(BROOD = d$BROOD[1], TICKS = sum(d$TICKS), n = nrow(d),
               HEIGHT = d$HEIGHT[1], YEAR = d$YEAR[1], LOCATION = d$LOCATION[1])))
ref$agg_TICKS <- agg$TICKS
ref$agg_n <- agg$n
mo <- glmer(TICKS ~ YEAR + scale(HEIGHT) + offset(log(n)) + (1|LOCATION),
            family = poisson, data = agg)
nbo <- glmer.nb(TICKS ~ YEAR + scale(HEIGHT) + offset(log(n)) + (1|LOCATION), data = agg)
ref$mo <- fitinfo(mo)
ref$nbo <- fitinfo(nbo)
ref$nbo$nb_theta <- getME(nbo, "glmer.nb.theta")
ref$anova_mo_nbo <- anovainfo(anova(mo, nbo))
nd <- data.frame(YEAR = factor(c("95", "96", "97"), levels = levels(agg$YEAR)),
                 HEIGHT = c(450, 450, 450), n = c(1, 1, 5))
ref$pred_mo <- unname(predict(mo, newdata = nd, re.form = NA, type = "response"))
ref$pred_nbo <- unname(predict(nbo, newdata = nd, re.form = NA, type = "response"))

## ---- Gamma GLMM with a log link (free dispersion) ---------------------------
gm <- glmer(Reaction ~ Days + (Days | Subject), data = sleepstudy,
            family = Gamma(link = "log"))
ref$gamma <- fitinfo(gm)
ref$gamma$ranef_Subject <- unname(as.matrix(ranef(gm)$Subject))

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/04_count_data_poisson_negative_binomial.json")
cat("done\n")
