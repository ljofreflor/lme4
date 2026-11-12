## R reference results for examples/02_crossed_and_nested_random_effects.ipynb
## Run from examples/:  Rscript r/02_crossed_and_nested_random_effects.R
suppressPackageStartupMessages({library(lme4); library(jsonlite)})

## summary of a fitted LMM, as a list
fitinfo <- function(m) {
    vc <- as.data.frame(VarCorr(m))
    list(REMLcrit = if (isREML(m)) REMLcrit(m) else NA,
         deviance = if (isREML(m)) NA else deviance(m),
         logLik = as.numeric(logLik(m)),
         theta = unname(getME(m, "theta")),
         theta_names = names(getME(m, "theta")),
         sigma = sigma(m),
         fixef = as.list(fixef(m)),
         se = unname(sqrt(diag(as.matrix(vcov(m))))),
         vc_grp = vc$grp, vc_sdcor = vc$sdcor, vc_vcov = vc$vcov,
         ngrps = as.list(ngrps(m)),
         ranef = lapply(ranef(m), function(r) list(levels = rownames(r), values = r[[1]])))
}
out <- list()

## ---------------------------------------------------------------- Penicillin
## ?Penicillin and lme4 book, ch. 2
pen_xtab <- xtabs(~ sample + plate, Penicillin)
out$penicillin_xtab_all_one <- all(pen_xtab == 1)
fm1 <- lmer(diameter ~ (1|plate) + (1|sample), Penicillin)
out$penicillin <- fitinfo(fm1)
out$penicillin_ML <- fitinfo(update(fm1, REML = FALSE))

## getME structure
Zt <- getME(fm1, "Zt"); Lt <- getME(fm1, "Lambdat"); L <- getME(fm1, "L")
out$penicillin_getME <- list(
    Zt_dim = dim(Zt), Zt_nnz = length(Zt@x), Zt_colsums = unname(Matrix::colSums(Zt)),
    Zt_rownames = rownames(Zt),
    Lambdat_dim = dim(Lt), Lambdat_diag = Matrix::diag(Lt), Lambdat_nnz = length(Lt@x),
    Lind = getME(fm1, "Lind"), Gp = getME(fm1, "Gp"),
    cnms = getME(fm1, "cnms"), l_i = as.list(getME(fm1, "l_i")),
    ## determinant of L (= sqrt det(Lambda'Z'Z Lambda + I)) is invariant to the fill-reducing permutation
    logdetL = as.numeric(2 * determinant(L, logarithm = TRUE, sqrt = TRUE)$modulus),
    ## dense crossproduct Lambda' Z' Z Lambda + I, also permutation-invariant
    LtZtZL_trace = sum(Matrix::diag(Matrix::tcrossprod(Lt %*% Zt))))

## ---------------------------------------------------------------- Pastes
## ?Pastes
out$pastes_isNested <- list(cask_in_batch = reformulas::isNested(Pastes$cask, Pastes$batch),
                            sample_in_batch = reformulas::isNested(Pastes$sample, Pastes$batch),
                            batch_in_sample = reformulas::isNested(Pastes$batch, Pastes$sample),
                            plate_in_sample = reformulas::isNested(Penicillin$plate, Penicillin$sample))
p1 <- lmer(strength ~ (1|batch) + (1|sample), Pastes)
p2 <- lmer(strength ~ (1|batch/cask), Pastes)
p3 <- lmer(strength ~ (1|batch) + (1|batch:cask), Pastes)
p4 <- lmer(strength ~ (1|batch/sample), Pastes)
p5 <- lmer(strength ~ (1|batch) + (1|cask), Pastes)
out$pastes <- list(fm1 = fitinfo(p1), fm2 = fitinfo(p2), fm3 = fitinfo(p3),
                   fm4 = fitinfo(p4), fm5 = fitinfo(p5))
out$pastes_head_ranef_fm4 <- list(levels = head(rownames(ranef(p4)[[1]])),
                                  values = head(ranef(p4)[[1]][[1]]))
out$pastes_head_ranef_fm1 <- list(levels = head(rownames(ranef(p1)[[1]])),
                                  values = head(ranef(p1)[[1]][[1]]))
## lme4 book, ch. 2: is the batch variance needed?  (ML fits + LRT)
p1ML  <- lmer(strength ~ 1 + (1|sample) + (1|batch), Pastes, REML = FALSE)
p1aML <- lmer(strength ~ 1 + (1|sample), Pastes, REML = FALSE)
aa <- anova(p1aML, p1ML)
out$pastes_anova <- list(npar = aa$npar, AIC = aa$AIC, BIC = aa$BIC, logLik = aa$logLik,
                         minus2logL = aa[["-2*log(L)"]], Chisq = aa$Chisq, Df = aa$Df,
                         p = aa[["Pr(>Chisq)"]])

## ---------------------------------------------------------------- InstEval
out$insteval_dim <- dim(InstEval)
out$insteval_xtab <- unname(as.vector(xtabs(~ service + dept, InstEval)))
tm <- system.time(fmI <- lmer(y ~ service + (1|s) + (1|d) + (1|dept:service), InstEval))
out$insteval_time <- tm[["elapsed"]]
out$insteval <- fitinfo(fmI)
out$insteval$feval <- fmI@optinfo$feval
out$insteval_isNested <- list(s_in_d = reformulas::isNested(InstEval$s, InstEval$d),
                              d_in_dept = reformulas::isNested(InstEval$d, InstEval$dept))
ZtI <- getME(fmI, "Zt")
out$insteval_getME <- list(Zt_dim = dim(ZtI), Zt_nnz = length(ZtI@x),
                           Lambdat_diag_unique = unique(Matrix::diag(getME(fmI, "Lambdat"))),
                           Gp = getME(fmI, "Gp"),
                           logdetL = as.numeric(2 * determinant(getME(fmI, "L"), logarithm = TRUE, sqrt = TRUE)$modulus))

out$R_version <- as.character(getRversion())
out$lme4_version <- as.character(packageVersion("lme4"))
dir.create("reference", showWarnings = FALSE)
write_json(out, "reference/02_crossed_and_nested_random_effects.json",
           digits = NA, auto_unbox = TRUE, pretty = TRUE)
