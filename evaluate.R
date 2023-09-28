source("evaluations.R")
source("methods.R")

library("optparse")
library("pbapply")

option_list = list(
  make_option(c("--datadir"), type="character", default="data", 
              help="data directory [default= %default]", metavar="character"),
  make_option(c("--resdir"), type="character", default="results", 
              help="results directory [default= %default]", metavar="character"),
  make_option(c("--outdir"), type="character", default="evaluations", 
              help="output directory [default= %default]", metavar="character"),
  make_option(c("--ncl"), type="integer", default=as.integer(1), 
              help="number of parallel process to use [default= %default]", metavar="integer"),
  make_option(c("--methods"), type="character", default="all", 
              help=paste0("methods to run [default= %default], one of:\n", 
                          paste0(names(methods), collapse = "\n")), metavar="character")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)


out_dir <- opt$outdir
res_dir <- opt$resdir
data_dir <- opt$datadir

if (opt$methods != "all"){
  mthds_names <- strsplit(gsub("", "", opt$methods), ",")[[1]]
  methods <- methods[mthds_names]
  message("running: ", names(methods))
}

allfiles <- list.files(path = data_dir, recursive = TRUE, pattern = "*.csv",  full.names = TRUE)


pboptions(type = "txt")
pblapply(allfiles, function(filename){
  
  name <- basename(filename)
  con <- file(filename)
  info <- readLines(con, 19)
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
  
  results <- lapply(c("oracle", names(methods)), function(nm){
    tryCatch(read.csv(file.path(res_dir, nm, name)), error = function(e)  NULL)
  })
  
  names(results) <- c("oracle", names(methods))
  
  evals <- lapply(results, function(res){
    if (!is.null(res)){
      model <- lm(y ~ x + res[,1])
      est <- list(z = res[,1], c = coef(model)[2], ci = confint(model)[2,])
    }

    sapply(evaluations, function(eval) {
      if (is.null(res)){
        return(NA)
      }
      eval(est, Z, x, y, U, args)
      })
  })
  
  for (nm in names(results)){
    dir.create(file.path(out_dir, nm), recursive = TRUE, showWarnings = FALSE)
    EE <- evals[[nm]]
    EE <- c(EE, args[c("ic", "causal_coeff", "dist", "noise", "latents",
                       "confounder", "proxy", "size", "noisesd", "distsd",
                       "independent")])
    EE$method <- nm
    write.csv(as.data.frame(EE), file = file.path(out_dir, nm, paste0("evals_", name)), row.names = FALSE)
    
  }

}, cl = opt$ncl)
