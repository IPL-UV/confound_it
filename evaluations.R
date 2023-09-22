source("methods.R")

evaluations <- list(
  cor = function(zest, z, x, y, proxy, args) abs(cor(zest, z[,args$ic])), 
  ae_cc_vs_ols = function(zest, z, x, y, proxy,  args){
    cest <- coef(lm(y ~ x + zest))[2]
    cols <- coef(lm(y ~ x))[2]
    return(unname(abs(cest - args$causal_coeff)) / unname(abs(cols - args$causal_coeff)))
  },
  ae_cc_vs_proxy = function(zest, z, x, y, proxy, args){
    cest <- coef(lm(y ~ x + zest))[2]
    cols <- coef(lm(y ~ x + as.matrix(proxy)))[2]
    return(unname(abs(cest - args$causal_coeff)) / unname(abs(cols - args$causal_coeff)))
  },
  ae_cc_vs_proxysmall = function(zest, z, x, y, proxy, args){
    cest <- coef(lm(y ~ x + zest))[2]
    cols <- coef(lm(y ~ x + as.matrix(proxy[,1:args$latents])))[2]
    return(unname(abs(cest - args$causal_coeff)) / unname(abs(cols - args$causal_coeff)))
  },
  ae_cc_vs_pca = function(zest, z, x, y, proxy, args){
    pca <- prcomp(proxy, rank. = args$latents)$x
    cest <- coef(lm(y ~ x + zest))[2]
    cols <- coef(lm(y ~ x + pca))[2]
    return(unname(abs(cest - args$causal_coeff)) / unname(abs(cols - args$causal_coeff)))
  },
  ae_cc = function(zest, z, x, y, proxy, args){
    cest <- coef(lm(y ~ x + zest))[2]
    return(unname(abs(cest - args$causal_coeff)))
  },
  are_cc = function(zest, z, x, y, proxy, args){
    cest <- coef(lm(y ~ x + zest))[2]
    return(unname(abs(cest - args$causal_coeff) / abs(args$causal_coeff)))
  },
  se_cc = function(zest, z, x, y, proxy, args){
    cest <- coef(lm(y ~ x + zest))[2]
    return(unname((cest - args$causal_coeff)^2))
  },
  pvals_naive = function(zest, z, x, y, proxy, args){
    get_pvals(zest, x, y)
  },
  pvals_adj = function(zest, z, x, y, proxy, args){
    return(RES(zest, proxy, x, y)[c("px", "py")])
  },
  conf_x = function(zest, z, x, y, proxy, args){
    tail(args$coefx, 1)
  },
  conf_y = function(zest, z, x, y, proxy, args){
    tail(args$coefy, 1)
  },
  ab_c = function(zest, z, x, y, proxy, args){
    a <- tail(args$coefx, 1)
    b <- args$causal_coeff
    c <- tail(args$coefy, 1)
    a*b + c
  },
  maxabs_coefx = function(zest, z, x, y, proxy, args){
    max(abs(args$coefx))
  },
  maxabs_coefy = function(zest, z, x, y, proxy, args){
    max(abs(args$coefy))
  },
  sumabs_coefx = function(zest, z, x, y, proxy, args){
    sum(abs(args$coefx))
  },
  sumabs_coefy = function(zest, z, x, y, proxy, args){
    sum(abs(args$coefy))
  }
)


