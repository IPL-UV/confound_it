source("methods.R")
source("evaluations.R")
library("reshape2")

res_dir <- "results"

data_dir <- "data"

allfiles <- list.files(path = data_dir, recursive = TRUE, pattern = "*.csv",  full.names = TRUE)

methods <- methods["optim_pval_ica"]
for (filename in allfiles){
  name <- basename(filename)
  
  
  con <- file(filename)
  info <- readLines(con, 18)
  parsed <- sapply(info, function(x) strsplit(gsub("# ", "", x), " = "))
  names(parsed) <- lapply(parsed, function(x) x[1])
  data <- read.csv(con, header = TRUE, comment.char = "#")
  
  parsed$outdir[2] <- paste0("\"", parsed$outdir[2], "\"")
  parsed$dist[2] <- paste0("\"", parsed$dist[2], "\"")
  parsed$noise[2] <- paste0("\"", parsed$noise[2], "\"")
  args <- lapply(parsed, function(x) eval(parse(text=x[2])))
  names(args) <- lapply(parsed, function(x) x[1])
  
  
  x <- data$X
  y <- data$Y
  U <- data[, grep("U.", names(data))]
  Z <- data[, grep("Z.", names(data))]
  
  results <- lapply(methods, function(meth){
    meth(x = x, y = y, proxy = U, rank = 20)
  })
  
  evals <- sapply(results, function(res){
    sapply(evaluations, function(eval) eval(res, Z, args))
  })
  ### save results:
  
  for (nm in names(results)){
    dir.create(file.path(res_dir, nm), recursive = TRUE, showWarnings = FALSE)
    
    write.csv(results[[nm]], file = file.path(res_dir, nm, name), row.names = FALSE)
  }
  
  EE <- reshape2::melt(evals, varnames = c("stats", "method"))
  EE <- cbind(EE, args[c("ic", "causal_coeff", "dist", "noise", "latents",
                         "confounder", "proxy", "size", "noisesd", "distsd",
                         "independent")])
  a <- tail(args$coefx, 1)
  b <- tail(args$coefy, 1)
  c <- args$causal_coeff
  EE$ab_c <- a*b - c
  
  write.csv(EE, file = file.path(res_dir, paste0("optim_pval_ica_", name)), row.names = FALSE)
}
