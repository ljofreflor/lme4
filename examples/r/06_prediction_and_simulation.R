## R reference for examples/06_prediction_and_simulation.ipynb
## Run (from examples/):  Rscript r/06_prediction_and_simulation.R
suppressPackageStartupMessages({library(lme4); library(jsonlite)})
out <- list()
un <- function(x) unname(as.vector(x))

## ---------------------------------------------------------------- LMM: sleepstudy
fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
out$fm1_fixef <- un(fixef(fm1))
out$fm1_vcov <- un(as.matrix(vcov(fm1)))
out$fm1_fitted <- un(predict(fm1))
out$fm1_pred_NA <- un(predict(fm1, re.form = NA))

pop <- data.frame(Days = 0:9)
out$pop_fit <- un(predict(fm1, newdata = pop, re.form = NA))
pop_se <- predict(fm1, newdata = pop, re.form = NA, se.fit = TRUE)
out$pop_se <- un(pop_se$se.fit)

subj <- expand.grid(Days = 0:9, Subject = factor(c("308", "309", "335")))
out$subj_fit <- un(predict(fm1, newdata = subj))
subj_se <- predict(fm1, newdata = subj, se.fit = TRUE)
out$subj_se <- un(subj_se$se.fit)

newsubj <- data.frame(Days = c(0, 5, 9, 0, 5, 9),
                      Subject = factor(c("308", "308", "308", "new", "new", "new")))
out$new_fit <- un(predict(fm1, newdata = newsubj, allow.new.levels = TRUE))
out$new_se <- un(predict(fm1, newdata = newsubj, allow.new.levels = TRUE, se.fit = TRUE)$se.fit)
out$new_error <- tryCatch({predict(fm1, newdata = newsubj); "no error"},
                          error = function(e) conditionMessage(e))
out$fm1_se_all <- un(predict(fm1, se.fit = TRUE)$se.fit)

## theoretical marginal (unconditional) variance of a new observation at Days = 0..9
vc <- VarCorr(fm1)$Subject
s2 <- sigma(fm1)^2
d <- 0:9
out$marg_var <- vc[1, 1] + 2 * d * vc[1, 2] + d^2 * vc[2, 2] + s2
out$sigma <- sigma(fm1)

## ---------------------------------------------------------------- crossed: Penicillin
fm3 <- lmer(diameter ~ 1 + (1 | plate) + (1 | sample), Penicillin)
out$pen_full <- un(predict(fm3))
out$pen_plate <- un(predict(fm3, re.form = ~ (1 | plate)))
out$pen_sample <- un(predict(fm3, re.form = ~ (1 | sample)))
out$pen_NA <- un(predict(fm3, re.form = NA))
out$pen_plate_se <- un(predict(fm3, re.form = ~ (1 | plate), se.fit = TRUE)$se.fit)

## ---------------------------------------------------------------- GLMM: cbpp
gm1 <- glmer(cbind(incidence, size - incidence) ~ period + (1 | herd),
             data = cbpp, family = binomial)
out$gm1_link <- un(predict(gm1))
out$gm1_resp <- un(predict(gm1, type = "response"))
pp <- data.frame(period = factor(1:4))
out$gm1_pop_link <- un(predict(gm1, newdata = pp, re.form = NA))
out$gm1_pop_resp <- un(predict(gm1, newdata = pp, re.form = NA, type = "response"))
se_l <- suppressWarnings(predict(gm1, newdata = pp, re.form = NA, se.fit = TRUE))
se_r <- suppressWarnings(predict(gm1, newdata = pp, re.form = NA, type = "response",
                                 se.fit = TRUE))
out$gm1_pop_se_link <- un(se_l$se.fit)
out$gm1_pop_se_resp <- un(se_r$se.fit)
hp <- expand.grid(period = factor(1:4), herd = factor(c("1", "5", "9")))
out$gm1_herd_resp <- un(predict(gm1, newdata = hp, type = "response"))
out$gm1_herd_se_resp <- un(suppressWarnings(
    predict(gm1, newdata = hp, type = "response", se.fit = TRUE))$se.fit)

## ---------------------------------------------------------------- simulate / bootMer
## (random: only structure and t0 are deterministic)
set.seed(101)
s1 <- simulate(fm1, nsim = 5)
out$sim_dim <- dim(s1)
s2c <- simulate(fm1, nsim = 2000, use.u = TRUE, seed = 1)
out$sim_cond_mean_err <- max(abs(rowMeans(s2c) - fitted(fm1)))
b1 <- bootMer(fm1, fixef, nsim = 200, seed = 101)
out$boot_t0 <- un(b1$t0)
out$boot_sd <- un(apply(b1$t, 2, sd))

## ---------------------------------------------------------------- power by simulation
## simulate() from a formula with assumed parameters
design <- expand.grid(Days = 0:9, Subject = factor(1:20))
pars <- list(beta = c(250, 3), theta = c(25/26, 0, 6/26), sigma = 26)
nsim <- 200
ysim <- simulate(~ Days + (Days | Subject), nsim = nsim, seed = 2024,
                 newdata = design, newparams = pars, family = gaussian)
tvals <- sapply(ysim, function(y) {
    design$Reaction <- y
    coef(summary(lmer(Reaction ~ Days + (Days | Subject), design)))["Days", "t value"]
})
out$power_nsim <- nsim
out$power_R <- mean(abs(tvals) > 2)
## approximate (Wald) power from the assumed model: SE(slope) for the balanced design
## Var(slope) = (sd_slope^2 + sigma^2 / sum((d - mean(d))^2)) / n_subjects
se_slope <- sqrt(((6)^2 + 26^2 / sum((0:9 - 4.5)^2)) / 20)
out$power_wald <- pnorm(3 / se_slope - 2) + pnorm(-3 / se_slope - 2)

writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/06_prediction_and_simulation.json")
cat("done\n")
