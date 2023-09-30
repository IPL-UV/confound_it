
get_pvals <- function(zz, x, y){
  m1 <- lm(x ~ zz) 
  m2 <- lm(y ~ zz + x)
  return(list(px = summary(m1)$coefficients[2,4], py = summary(m2)$coefficients[2,4]))
}


optim_res<- function(x, y, proxy, rank = 10, ica = FALSE){
  
  x <- scale(x)
  y <- scale(y)
  ## define objective function
  fopt <- function(w, UU){
    w <- w / sqrt(sum(w^2))
    zz <- UU %*% w
    m1 <- lm(x ~ zz) 
    m2 <- lm(y ~ zz + x)
    sum(residuals(m1)) + sum(residuals(m2))
  }
  
  if (ica){
    Ureduced <- fastICA::fastICA(proxy, n.comp = rank)$S
  } else {
    Ureduced <- prcomp(proxy, rank. = rank, scale. = TRUE)$x
  }
  
  ## initial point
  win <- rep(0.5, ncol(Ureduced))
  
  ## brute force optimization
  res <- optim(win, fopt, UU = Ureduced)
  
  west <- res$par / sqrt(sum(res$par^2))
  
  ## compute estimated latent
  zest <- Ureduced %*% west
  
  return(zest[,1])
}


selnaive_pca <- function(x, y, proxy, rank = 10, ...){
  Ureduced <- prcomp(proxy, rank. = 10, ...)$x
  
  
  pvals <- sapply(seq_len(ncol(Ureduced)), function(i) sum(unlist(get_pvals(Ureduced[,i], x, y))))
  
  imin <- which.min(pvals)
  
  zest <- Ureduced[,imin]
  return(zest)
}



selnaive_ica <- function(x, y, proxy, rank = 10, ...){
  Ureduced <- fastICA::fastICA(proxy, n.comp = rank, ...)$S
  
  pvals <- sapply(seq_len(ncol(Ureduced)), function(i) sum(unlist(get_pvals(Ureduced[,i], x, y))))
  
  imin <- which.min(pvals)
  
  zest <- Ureduced[,imin]
  return(zest)
}


selnaive_pls <- function(x, y, proxy, rank = 10, ...){
  res <- mdatools::pls.run(x = proxy, y = cbind(x,y), ncomp = rank)
  est <- res$xscores
  
  pvals <- sapply(seq_len(ncol(est)), function(i) sum(unlist(get_pvals(est[,i], x, y))))
  
  imin <- which.min(pvals)
  return(unname(est[,imin]))
}

opls_y <- function(x, y, proxy, ...){
  res <- ropls::opls(x = proxy, y = y, predI = 1, orthoI = 1,
                     fig.pdfC = "none", info.txtC = "none")
  est <- res@scoreMN
  return(est[,1])
}


methods <- list(
  pls_sel_naive = selnaive_pls,
  ica_sel_naive = selnaive_ica,
  pca_sel_naive = selnaive_pca#,
  #optim_res = function(x, y, proxy, rank = 10) optim_res(x, y, proxy = proxy,
  #                                                         rank = rank,
  #                                                         ica = FALSE)
)
