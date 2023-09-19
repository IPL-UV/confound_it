
evaluations <- list(
  cor = function(zest, z, x, y, args) abs(cor(zest, z[,args$ic])), 
  ae_cc_norm = function(zest, z, x, y, args){
    cest <- coef(lm(y ~ x + zest))[2]
    cols <- coef(lm(y ~ x))[2]
    return(unname(abs(cest - args$causal_coeff)) / abs(cols - args$causal_coeff))
  },
  ae_cc_norm = function(zest, z, x, y, args){
    cest <- coef(lm(y ~ x + zest))[2]
    return(unname(abs(cest - args$causal_coeff)))
  },
  are_cc = function(zest, z, x, y, args){
    cest <- coef(lm(y ~ x + zest))[2]
    return(unname(abs(cest - args$causal_coeff) / abs(args$causal_coeff)))
  },
  se_cc = function(zest, z, x, y, args){
    cest <- coef(lm(y ~ x + zest))[2]
    return(unname((cest - args$causal_coeff)^2))
  }
)


