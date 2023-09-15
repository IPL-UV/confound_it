
evaluations <- list(
  cor = function(zest, z, args) cor(zest, z[,args$ic]), 
  mse = function(zest, z, args) mean((scale(zest) - sign(cor(zest, z[,args$ic]))*scale(z[,args$ic]))^2),
  mae = function(zest, z, args) mean(abs(scale(zest) - sign(cor(zest, z[,args$ic]))*scale(z[,args$ic])))
)


