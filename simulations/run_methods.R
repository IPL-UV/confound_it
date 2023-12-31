source("methods.R")
source("evaluations.R")
library("reshape2")
library("pbapply")

library("optparse")

option_list = list(
  make_option(c("--datadir"), type="character", default="data", 
              help="data directory [default= %default]", metavar="character"),
  make_option(c("--outdir"), type="character", default="results", 
              help="output directory [default= %default]", metavar="character"),
  make_option(c("--ncl"), type="integer", default=as.integer(1), 
              help="number of parallel process to use [default= %default]", metavar="integer"),
  make_option(c("--methods"), type="character", default="all", 
              help=paste0("methods to run [default= %default], one of:\n", 
                          paste0(names(methods), collapse = "\n")), metavar="character"),
  make_option("--oracle", action = "store_true", default = FALSE, help = "whether to run oracle")
  
  )

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)


res_dir <- opt$outdir

data_dir <- opt$datadir

allfiles <- list.files(path = data_dir, recursive = TRUE, pattern = "*.csv",  full.names = TRUE)

if (opt$methods != "all"){
  mthds_names <- strsplit(gsub(" ", "", opt$methods), ",")[[1]]
  methods <- methods[mthds_names]
}

message("running: \n", paste(names(methods), collapse = "\n"))
if (opt$oracle){
  message("oracle")
}

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
  
  system.time(results <- lapply(methods, function(meth){
    tryCatch(meth(x = x, y = y, proxy = U, rank = args$latents), error = function(e) NULL)
  }))
  
  if (opt$oracle){
    results$oracle <- c(Z[,args$ic])
  }
  
  ### save results:
  for (nm in names(results)){
    dir.create(file.path(res_dir, nm), recursive = TRUE, showWarnings = FALSE)
    write.csv(results[[nm]], file = file.path(res_dir, nm, name),
              row.names = FALSE, )
  }
  return(NULL)
}, cl = NULL) #opt$ncl)

