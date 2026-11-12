## R reference for examples/05_nonlinear_mixed_models.ipynb
## Run (from examples/):  Rscript r/05_nonlinear_mixed_models.R
## Also exports the base-R datasets Orange and Theoph to data/*.csv.
suppressPackageStartupMessages({library(lme4); library(jsonlite)})

ref <- list(R_version = R.version.string, lme4_version = as.character(packageVersion("lme4")))

## ---- data (package 'datasets') -------------------------------------------
write.csv(Orange, "data/Orange.csv", row.names = FALSE)
write.csv(Theoph, "data/Theoph.csv", row.names = FALSE)
ref$levels <- list(Tree = levels(Orange$Tree), Subject = levels(Theoph$Subject))

## The deviance function of the fit, evaluated afresh at the estimates, with the
## penalized nonlinear least squares iterations started from u = 0 (see notebook).
devAtOpt <- function(m) {
    df <- update(m, devFunOnly = TRUE)
    rho <- environment(df)
    rho$u0[] <- 0
    p <- getME(m, "theta")
    if (m@devcomp$dims[["nAGQ"]] > 0) p <- c(p, fixef(m))
    df(p)
}
fitinfo <- function(m) {
    vc <- as.data.frame(VarCorr(m))
    list(fixef = as.list(fixef(m)),
         theta = unname(getME(m, "theta")),
         logLik = as.numeric(logLik(m)),
         df = attr(logLik(m), "df"),
         deviance = deviance(m),
         AIC = AIC(m), BIC = BIC(m),
         sigma = sigma(m),
         se = unname(sqrt(diag(as.matrix(vcov(m))))),
         vcov = unname(as.matrix(vcov(m))),
         se_RX = unname(sqrt(diag(as.matrix(vcov(m, use.hessian = FALSE))))),
         devAtOpt = devAtOpt(m),
         sdcor = vc$sdcor,
         grp = vc$grp, var1 = vc$var1)
}

## ---- ?nlmer: Orange trees, SSlogis -----------------------------------------
startvec <- c(Asym = 200, xmid = 725, scal = 350)
nm1 <- nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym|Tree,
             Orange, start = startvec)
ref$nm1 <- fitinfo(nm1)
ref$nm1$ranef <- ranef(nm1)$Tree$Asym
ref$nm1$condsd <- sqrt(attr(ranef(nm1, condVar = TRUE)$Tree, "postVar")[1, 1, ])
ref$nm1$coef <- as.list(coef(nm1)$Tree)
ref$nm1$fitted <- unname(fitted(nm1))
ref$nm1$residuals <- unname(residuals(nm1))
ref$nm1$predict_NA <- unname(predict(nm1, re.form = NA))

## nAGQ = 0
nm1a <- update(nm1, nAGQ = 0L)
ref$nm1a <- fitinfo(nm1a)

## user-built function from deriv()
nform <- ~Asym/(1+exp((xmid-input)/scal))
nfun <- deriv(nform, namevec = c("Asym","xmid","scal"),
              function.arg = c("input","Asym","xmid","scal"))
nm1b <- update(nm1, circumference ~ nfun(age, Asym, xmid, scal) ~ Asym | Tree)
ref$nm1b <- fitinfo(nm1b)
ref$nfun_grad <- unname(attr(nfun(2, 1, 3, 4), "gradient"))

## user-built function with hand-written gradient
nfun2 <- function(input, Asym, xmid, scal) {
    value <- Asym/(1+exp((xmid-input)/scal))
    grad <- cbind(Asym=1/(1+exp((xmid-input)/scal)),
              xmid=-Asym/(1+exp((xmid-input)/scal))^2*1/scal*
                    exp((xmid-input)/scal),
              scal=-Asym/(1+exp((xmid-input)/scal))^2*
                     -(xmid-input)/scal^2*exp((xmid-input)/scal))
    attr(value,"gradient") <- grad
    value
}
nm1c <- update(nm1, circumference ~ nfun2(age, Asym, xmid, scal) ~ Asym | Tree)
ref$nm1c <- fitinfo(nm1c)

## random effects on more parameters
nm1d <- nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid | Tree,
              Orange, start = startvec)
ref$nm1d <- fitinfo(nm1d)
ref$nm1d$ranef <- as.list(ranef(nm1d)$Tree)
nm1u <- nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ (Asym|Tree) + (xmid|Tree),
              Orange, start = startvec)
ref$nm1u <- fitinfo(nm1u)
nm1v <- nlmer(circumference ~ SSlogis(age, Asym, xmid, scal) ~ Asym + xmid || Tree,
              Orange, start = startvec)
ref$nm1v <- fitinfo(nm1v)

## likelihood-ratio statistics (anova() does not support nlmerMod objects)
ref$lrt <- list(d_vs_1 = 2 * (as.numeric(logLik(nm1d)) - as.numeric(logLik(nm1))))

## fitted curves per tree from coef()
grid <- seq(100, 1600, by = 50)
cf <- coef(nm1)$Tree
ref$curves <- lapply(split(cf, rownames(cf)), function(p)
    unname(SSlogis(grid, p$Asym, p$xmid, p$scal)))

## ---- tests/nlmer.R: Theoph, SSfol -----------------------------------------
Th.start <- c(lKe = -2.5, lKa = 0.5, lCl = -3)
nm2 <- nlmer(conc ~ SSfol(Dose, Time,lKe, lKa, lCl) ~ (lKe+lKa+lCl|Subject),
             Theoph, start = Th.start, control = nlmerControl(tolPwrss = 1e-8))
ref$nm2 <- fitinfo(nm2)
nm3 <- nlmer(conc ~ SSfol(Dose, Time,lKe, lKa, lCl) ~
                 (lKe|Subject) + (lKa|Subject) + (lCl|Subject),
             Theoph, start = Th.start)
ref$nm3 <- fitinfo(nm3)
nm4 <- nlmer(conc ~ SSfol(Dose, Time,lKe, lKa, lCl) ~ (lKa+lCl|Subject),
             Theoph, start = Th.start, control = nlmerControl(tolPwrss = 1e-8))
ref$nm4 <- fitinfo(nm4)
nm5 <- nlmer(conc ~ SSfol(Dose, Time,lKe, lKa, lCl) ~ (lKa|Subject) + (lCl|Subject),
             Theoph, start = Th.start, control = nlmerControl(tolPwrss = 1e-8))
ref$nm5 <- fitinfo(nm5)
ref$nm5$coef <- as.list(coef(nm5)$Subject)
ref$nm5$fitted <- unname(fitted(nm5))

writeLines(toJSON(ref, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "reference/05_nonlinear_mixed_models.json")
cat("wrote reference/05_nonlinear_mixed_models.json\n")
