source("methods.R")

evaluations <- list(
  cor = function(est, z, x, y, proxy, args) abs(cor(est$z, z[,args$ic])), 
  ae_cc_vs_ols = function(est, z, x, y, proxy, args){
    c1 <- coef(lm(y ~ x))[2]
    return(unname(abs(est$c - args$causal_coeff)) / 
             unname(abs(c1 - args$causal_coeff)))
  },
  ae_cc_vs_proxy = function(est, z, x, y, proxy, args){
    c1 <- coef(lm(y ~ x + as.matrix(proxy)))[2]
    return(unname(abs(est$c - args$causal_coeff)) / unname(abs(c1 - args$causal_coeff)))
  },
  ae_cc_vs_pca = function(est, z, x, y, proxy, args){
    pca <- prcomp(proxy, rank. = args$latents)$x
    c1 <- coef(lm(y ~ x + pca))[2]
    return(unname(abs(est$c - args$causal_coeff)) / unname(abs(c1 - args$causal_coeff)))
  },
  ae_cc_vs_pearlpca = function(est, z, x, y, proxy, args){
    pca <- prcomp(proxy, rank. = 2)$x
    v <- pca[,1]
    w <- pca[,2]
    # https://arxiv.org/pdf/1203.3504.pdf Eq (12)
    c0 <- (cov(x, y) * cov(x, v) - cov(y, w) * cov(w, v)) / 
      (cov(x, w) * var(x) - cov(x, w) * cov(w, v))
    return(unname(abs(est$c - args$causal_coeff)) / unname(abs(c0 - args$causal_coeff)))
  },
  ae_cc_vs_pearlfirst = function(est, z, x, y, proxy, args){
    v <- proxy[,1]
    w <- proxy[,2]
    # https://arxiv.org/pdf/1203.3504.pdf Eq (12)
    c0 <- (cov(x, y) * cov(x, v) - cov(y, w) * cov(w, v)) / 
      (cov(x, w) * var(x) - cov(x, w) * cov(w, v))
    return(unname(abs(est$c - args$causal_coeff)) / unname(abs(c0 - args$causal_coeff)))
  },
  ae_cc = function(est, z, x, y, proxy, args){
    return(unname(abs(est$c - args$causal_coeff)))
  },
  are_cc = function(est, z, x, y, proxy, args){
    return(unname(abs(est$c - args$causal_coeff) / abs(args$causal_coeff)))
  },
  se_cc = function(est, z, x, y, proxy, args){
    return(unname((est$c - args$causal_coeff)^2))
  },
  pvals_naive = function(est, z, x, y, proxy, args){
    get_pvals(est$z, x, y)
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


