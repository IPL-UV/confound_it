
get_pvals <- function(zz, x, y, method = "sum"){
  m1 <- lm(x ~ zz) 
  m2 <- lm(y ~ zz + x)
  return(list(px = summary(m1)$coefficients[2,4], py = summary(m2)$coefficients[2,4]))
}


RES <- function(zz, UU, x, y){
  UU <- scale(UU, scale = FALSE)
  y <- scale(y)
  x <- scale(x)
  zz <- scale(zz)
  my1 <- lm(y ~ x + UU)
  mx1 <- lm(x ~ UU)
  a1 <- sum(residuals(my1)^2)
  b1 <- sum(residuals(mx1)^2)
  
  U1 <- qr.Q(qr(cbind(zz, UU)))[,-1] ## hoping pivoting is not used
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

selnaive_pca <- function(x, y, proxy, rank = 10, ...){
  Ureduced <- prcomp(proxy, rank. = 10, ...)$x
  
  
  pvals <- sapply(seq_len(ncol(Ureduced)), function(i) sum(unlist(get_pvals(Ureduced[,i], x, y))))
  
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

selnaive_ica <- function(x, y, proxy, rank = 10, ...){
  Ureduced <- fastICA::fastICA(proxy, n.comp = rank, ...)$S
  
  pvals <- sapply(seq_len(ncol(Ureduced)), function(i) sum(unlist(get_pvals(Ureduced[,i], x, y))))
  
  imin <- which.min(pvals)
  
  zest <- Ureduced[,imin]
  return(zest)
}


pls_1 <- function(x, y, proxy, rank = 10, ...){
  res <- ropls::opls(x = proxy, y = cbind(x,y), predI = rank,
                     fig.pdfC = "none", info.txtC = "none")
  est <- res@scoreMN
  return(unname(est[,1]))
}

sel_pls <- function(x, y, proxy, rank = 10, ...){
  res <- ropls::opls(x = proxy, y = cbind(x,y), predI = rank,
                     fig.pdfC = "none", info.txtC = "none")
  est <- res@scoreMN
  res <- lapply(seq_len(ncol(est)), function(i){
    RES(est[,i], est, x, y)
  })
  
  pvals <- sapply(res, function(rr) rr$px + rr$py)
  
  imin <- which.min(pvals)
  return(unname(est[,imin]))
}

selnaive_pls <- function(x, y, proxy, rank = 10, ...){
  res <- ropls::opls(x = proxy, y = cbind(x,y), predI = rank,
                     fig.pdfC = "none", info.txtC = "none")
  est <- res@scoreMN
  
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
  #pca1 = function(x, y, proxy, rank, ...){return(prcomp(proxy, rank. = 1, ...)$x)},
  #pls1 = pls_1,
  #opls_y = opls_y,
  #pls_sel = sel_pls,
  pls_sel_naive = selnaive_pls,
  #ica_sel = sel_ica,
  ica_sel_naive = selnaive_ica,
  pca_sel_naive = selnaive_pca#,
  #optim_pval = function(x, y, proxy, rank = 10) optim_pval(x, y, proxy = proxy,
  #                                                         rank = rank,
  #                                                         ica = FALSE),
  #optim_pval_ica = function(x, y, proxy, rank = 10) optim_pval(x, y, 
  #                                                             proxy = proxy,
  #                                                             rank = rank,
  #                                                             ica = TRUE)
)
