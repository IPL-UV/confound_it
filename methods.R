RES <- function(zz, UU, x, y){
  UU <- scale(UU, scale = FALSE)
  y <- scale(y)
  x <- scale(x)
  zz <- scale(zz)
  my1 <- lm(y ~ x + UU)
  mx1 <- lm(x ~ UU)
  a1 <- sum(residuals(my1)^2)
  b1 <- sum(residuals(mx1)^2)
  
  U1 <- qr.Q(qr(cbind(zz, UU)))[,2:ncol(UU)] ## hoping pivoting is not used
  my2 <- lm(y ~ x + U1)
  a2 <-  sum(residuals(my2)^2)
  mx2 <- lm(x ~ U1)
  b2 <- sum(residuals(mx2)^2)
  
  ##1 compute the two F-statistics for
  ## (my1 vs my2) and for (mx1 vs mx2)
  fy = (a2 - a1) / a1
  fx = (b2 - b1) / b1
  
  ## compute the p-values
  py  = pf(fy * (nrow(UU) - ncol(UU) - 1) ,
           df1 = 1, df2 =  nrow(UU) - ncol(UU) - 1, lower.tail = FALSE)
  px  = pf(fx * (nrow(UU) - ncol(UU)) ,
           df1 = 1, df2 = nrow(UU) - ncol(UU), lower.tail = FALSE)
  
  return(list(a = a2, b = b2, fy = fy, fx = fx, py = py ,  px = px))
}



optim_pval <- function(x, y, proxy, rank = 10, ica = FALSE){
  
  ## define objective function
  fopt <- function(w, UU){
    w <- w / sqrt(sum(w^2))
    zz <- UU %*% w
    rr <- RES(zz, UU, x, y)
    log(rr$px + rr$py)
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


sel_pca <- function(x, y, proxy, rank = 10, ...){
  Ureduced <- prcomp(proxy, rank. = 10, ...)$x
  
  res <- lapply(seq_len(ncol(Ureduced)), function(i){
    RES(Ureduced[,i], Ureduced, x, y)
  })
  
  pvals <- sapply(res, function(rr) rr$px + rr$py)
  
  imin <- which.min(pvals)
  
  zest <- Ureduced[,imin]
  return(zest)
}


sel_ica <- function(x, y, proxy, rank = 10, ...){
  Ureduced <- fastICA::fastICA(proxy, n.comp = rank, ...)$S
  
  res <- lapply(seq_len(ncol(Ureduced)), function(i){
    RES(Ureduced[,i], Ureduced, x, y)
  })
  
  pvals <- sapply(res, function(rr) rr$px + rr$py)
  
  imin <- which.min(pvals)
  
  zest <- Ureduced[,imin]
  return(zest)
}


methods <- list(
  sel_ica = sel_ica,
  sel_pca = sel_pca,
  optim_pval = function(x, y, proxy, rank = 10) optim_pval(x, y, proxy = proxy,
                                                           rank = rank,
                                                           ica = FALSE),
  optim_pval_ica = function(x, y, proxy, rank = 10) optim_pval(x, y, 
                                                               proxy = proxy,
                                                               rank = rank,
                                                               ica = TRUE)
)