library("optparse")

option_list = list(
  make_option(c("--rep"), type = "integer", default = 1,
              help=paste0("number of replications [default= %default]")),
  make_option(c("--latents"), type = "integer", default = 4,
              help=paste0("the number of latents [default= %default]")),
  make_option(c("--confounder"), type = "integer", default = 1,
              help=paste0("the number of confounding latents [default= %default]")),
  make_option(c("--proxy"), type = "integer", default = 100,
              help=paste0("the number of proxys [default= %default]"), 
              metavar="integer"),
  make_option(c("--dist"), type="character", default="gaussian, exponential, gamma, uniform",
              help=paste0("the distributuion(s) of the latents [default= %default]"), 
              metavar="character"),
  make_option(c("--distsd"), type = "double", default = 1,
              help = "the sd of the latent dist (should be approx the sd in all cases), 
                     [default=%default]"),
  make_option(c("--noise"), type="character", default="gaussian, exponential, gamma, uniform",
              help=paste0("the distributuion(s) of the additive noise [default= %default]"), 
              metavar="character"),
  make_option(c("--noisesd"), type = "double", default = 1,
              help = "the strength of the noise (should be approx the sd in all cases), 
                     [default=%default]"),
  make_option(c("--independent"), action = "store_true", default = FALSE,
              help = "if latents should be independent [default= %default]"),
  make_option(c("--noiseproxy"), type = "double", default = 0.0,
              help = "noise added in the proxy [default= %default]"),
  make_option(c("--outdir"), type="character", default="data", 
              help="output directory [default= %default]", metavar="character"),
  make_option(c("--ncl"), type="integer", default=as.integer(1), 
              help="number of parallel process to use [default= %default]", metavar="integer"),
  make_option(c("--size"), type="integer", default=1000, 
              help="sample sizes [default= %default]", metavar="character")
  
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)


## simulate a toy model
N <- opt$size

## noises
noise <- gsub(" ", "", strsplit(opt$noise, ",")[[1]], fixed = TRUE)

## latent dist 
dist <- gsub(" ", "", strsplit(opt$dist, ",")[[1]], fixed = TRUE)


## number of paralllel process to use
ncl <- opt$ncl

## timestamp
tmstmp <- format(Sys.time(), "%H:%M:%S_%d%m%Y")

## exp_dir 
exp_dir <- file.path(opt$outdir)
dir.create(exp_dir, showWarnings = FALSE, recursive = TRUE)

cat("\n ------------------- \n ", file = file.path(exp_dir, "log.txt"),
    append = TRUE)
cat(tmstmp, file = file.path(exp_dir, "log.txt"), append = TRUE)
cat("Arguments:\n", file = file.path(exp_dir, "log.txt"), append = TRUE)
cat(paste(names(opt), as.character(opt), collapse = "\n"),
    file = file.path(exp_dir, "log.txt"), append = TRUE)


## expand options
options <- opt
options$noise <- noise
options$dist <- dist
options$help <- NULL
options <- expand.grid(options)

for (i in seq_len(nrow(options))){
  oo <- options[i, ,drop = FALSE]
  for (r in seq_len(oo$rep)){
    ## define (independent) latents
    distfun <- switch (as.character(oo$dist),
                       gaussian = function(n, s){rnorm(n,0,s)},
                       exponential = function(n, s){rexp(n, 1/s)},
                       gamma = function(n, s){rgamma(n, shape = 1, scale = s)},
                       uniform = function(n, s){runif(n, min = -s*sqrt(3), max = s*sqrt(3))}
    )
    noisefun <- switch (as.character(oo$noise),
                        gaussian = function(n, s){rnorm(n,0,s)},
                        exponential = function(n, s){rexp(n, 1/s)},
                        gamma = function(n, s){rgamma(n, shape = 1, scale = s)},
                        uniform = function(n, s){runif(n, min = -s*sqrt(3), max = s*sqrt(3))}
    )
    latents <- oo$latents
    size <- oo$size
    Z <- matrix(distfun(size * latents, oo$distsd), ncol = latents)
    if (!oo$independent){
      LL <- matrix(data = rnorm(latents * latents), ncol = latents)
      Z <-  Z %*% LL
    }
    
    ## define proxy as random mixing of latents
    nproxy <- oo$proxy
    A <- matrix(rnorm(latents * nproxy), nrow = latents, ncol = nproxy)
    U <- Z %*% A + rnorm(nrow(Z) * ncol(A), 0, sd = oo$noiseproxy)
    
    ### the causal system X --> Y
    truecausal <- runif(1, min = 0.5, max = 1.5)
    freelatents <- latents - oo$confounder
    ix <- sample(freelatents, freelatents / 3) ## latents affecting x
    iy <- sample((1:freelatents)[-ix], freelatents / 3) ## latents affecting y
    coefx <- runif(length(ix) + oo$confounder, min = 0.5, max = +1.5)
    coefy <- runif(length(iy) + oo$confounder, min = 0.5, max = +1.5)
    ic <- tail(seq_len(latents), oo$confounder)
    ## define X -> Y , with latents Z[,ic] as confounder
    x <- Z[,c(ix, ic), drop = FALSE] %*% coefx + noisefun(N, oo$noisesd)
    y <- truecausal * x + Z[,c(iy, ic), drop = FALSE] %*% coefy + noisefun(N, oo$noisesd)

    
    
    
    ### saving things
    oo_p <- lapply(oo, function(x) as.character(x))
    oo_p$ix <- ix
    oo_p$iy <- iy
    oo_p$ic <- ic
    oo_p$causal_coeff <- truecausal
    oo_p$coefx <- coefx
    oo_p$coefy <- coefy
    dat <- data.frame(Z = Z, U = U, X = x, Y = y)
    filename <- file.path(exp_dir, paste0("rep",r,"_", oo$noise, oo$noisesd,
                                          oo$dist, oo$distsd,
                                          ifelse(oo$independent,
                                                 "_independent", 
                                                 "_dependent"), 
                                          "_", oo$latents, "_",
                                          oo$confounder,
                                          "_", oo$proxy,
                                          "_", oo$size, ".csv"))
    con <- file(filename, open = "wt")
    writeLines(paste("#", names(oo_p),"=",as.character(oo_p)), con = con)
    write.csv(dat, con, row.names = FALSE)
    close(con)
  }
}
