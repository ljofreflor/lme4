## R reference for examples/03_binomial_glmm.ipynb
## Run (from examples/):  Rscript r/03_binomial_glmm.R
suppressPackageStartupMessages({library(lme4); library(jsonlite)})

ref <- list(R_version = R.version.string, lme4_version = as.character(packageVersion("lme4")))

fitinfo <- function(m) {
    out <- list(fixef = as.list(fixef(m)),
                theta = unname(getME(m, "theta")),
                logLik = as.numeric(logLik(m)),
                deviance = deviance(m),
                AIC = AIC(m), BIC = BIC(m),
                se = unname(sqrt(diag(as.matrix(vcov(m))))),
                vcov = unname(as.matrix(vcov(m))),
                messages = m@optinfo$conv$lme4$messages)
    if (is.null(out$messages)) out$messages <- character(0)
    out
}

## ---- cbpp: matrix response (?glmer, ?cbpp) --------------------------------
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
ref$gm1 <- fitinfo(gm1)
ref$gm1$ranef <- ranef(gm1)$herd[, 1]
ref$gm1$condsd <- sqrt(attr(ranef(gm1, condVar = TRUE)$herd, "postVar")[1, 1, ])
ref$gm1$sdcor <- as.data.frame(VarCorr(gm1))$sdcor

## proportion + weights
m1p <- glmer(incidence / size ~ period + (1 | herd), weights = size,
             family = binomial, data = cbpp)
ref$m1p <- fitinfo(m1p)
ref$m1p$ranef <- ranef(m1p)$herd[, 1]

## ---- nAGQ: 0, 1, 9, 25 -----------------------------------------------------
ref$nAGQ <- list()
for (q in c(0L, 1L, 9L, 25L)) {
    m <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               data = cbpp, family = binomial, nAGQ = q)
    ref$nAGQ[[paste0("q", q)]] <- fitinfo(m)
}

## ---- link functions ----------------------------------------------------------
ref$links <- list()
for (lk in c("logit", "probit", "cloglog")) {
    m <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
               data = cbpp, family = binomial(link = lk))
    ref$links[[lk]] <- fitinfo(m)
}

## ---- observation-level random effect ---------------------------------------
cbpp$obs <- factor(1:nrow(cbpp))
gm2 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd) + (1 | obs),
             family = binomial, data = cbpp)
ref$gm2 <- fitinfo(gm2)
ref$gm2$sdcor <- as.data.frame(VarCorr(gm2))$sdcor
a <- anova(gm1, gm2)
ref$anova12 <- list(npar = a$npar, AIC = a$AIC, BIC = a$BIC, logLik = a$logLik,
                    deviance = a$deviance, Chisq = a$Chisq[2], Df = a$Df[2],
                    p = a[["Pr(>Chisq)"]][2])
## Pearson overdispersion ratio for gm1 (GLMM FAQ)
rp <- residuals(gm1, type = "pearson")
ref$overdisp <- list(chisq = sum(rp^2), rdf = df.residual(gm1),
                     ratio = sum(rp^2) / df.residual(gm1),
                     p = pchisq(sum(rp^2), df = df.residual(gm1), lower.tail = FALSE))

## ---- glmer and glm log-likelihoods are consistent ---------------------------
gm1Devfun <- update(gm1, devFunOnly = TRUE)
gm0 <- glm(cbind(incidence, size - incidence) ~ period, family = binomial, data = cbpp)
ref$glm0 <- list(coef = unname(coef(gm0)), m2logLik = c(-2 * logLik(gm0)),
                 devfun_at_glm = gm1Devfun(c(0, coef(gm0))))

## ---- odds ratios with Wald CIs -----------------------------------------------
ci <- confint(gm1, method = "Wald")
ref$wald <- list(rownames = rownames(ci), lower = unname(ci[, 1]), upper = unname(ci[, 2]))
ref$OR <- list(est = unname(exp(fixef(gm1))),
               lower = unname(exp(ci[-1, 1])), upper = unname(exp(ci[-1, 2])))
cc <- coef(summary(gm1))
ref$coeftab <- list(z = unname(cc[, "z value"]), p = unname(cc[, "Pr(>|z|)"]))

## ---- predicted probabilities ------------------------------------------------
ref$pred <- list(
    resp = unname(predict(gm1, type = "response")),
    link = unname(predict(gm1)),
    fitted = unname(fitted(gm1)),
    pop = unname(predict(gm1, newdata = data.frame(period = factor(1:4)),
                         re.form = NA, type = "response")),
    herd1 = unname(predict(gm1, newdata = data.frame(period = factor(1:4), herd = "1"),
                           type = "response")))

## ---- VerbAgg: crossed random effects ------------------------------------------
fmVA <- glmer(r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item),
              family = binomial, data = VerbAgg)
ref$VerbAgg <- fitinfo(fmVA)
ref$VerbAgg$fixef_names <- names(fixef(fmVA))
ref$VerbAgg$sdcor <- as.data.frame(VarCorr(fmVA))$sdcor
ref$VerbAgg$item_ranef <- ranef(fmVA)$item[, 1]
fmVA0 <- glmer(r2 ~ Anger + Gender + btype + situ + (1 | id) + (1 | item),
               family = binomial, data = VerbAgg, nAGQ = 0L)
ref$VerbAgg0 <- fitinfo(fmVA0)

## ---- toenail: longitudinal binary outcome -------------------------------------
toemod <- glmer(outcome ~ time * treatment + (1 | patientID),
                data = toenail, family = binomial(link = "logit"))
ref$toenail <- fitinfo(toemod)
ref$toenail$fixef_names <- names(fixef(toemod))
toemod20 <- glmer(outcome ~ time * treatment + (1 | patientID),
                  data = toenail, family = binomial, nAGQ = 20)
ref$toenail20 <- fitinfo(toemod20)
ref$toenail_ttab <- as.list(table(table(toenail$patientID)))

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/03_binomial_glmm.json")
cat("done\n")
