source("methods.R")
source("evaluations.R")

res_dir <- "results"

data_dir <- "data"

allfiles <- list.files(path = data_dir, recursive = TRUE, pattern = "*.csv",  full.names = TRUE)


for (filename in allfiles){
  name <- basename(filename)
  
  
  con <- file(filename)
  info <- readLines(con, 17)
  parsed <- sapply(info, function(x) strsplit(gsub("# ", "", x), " = "))
  names(parsed) <- lapply(parsed, function(x) x[1])
  data <- read.csv(con, header = TRUE, comment.char = "#")
  
  parsed$outdir[2] <- paste0("\"", parsed$outdir[2], "\"")
  parsed$dist[2] <- paste0("\"", parsed$dist[2], "\"")
  parsed$noise[2] <- paste0("\"", parsed$noise[2], "\"")
  args <- lapply(parsed, function(x) eval(parse(text=x[2])))
  names(args) <- lapply(parsed, function(x) x[1])
  
  
  methods <- list(
    sel_ica = sel_ica,
    sel_pca = sel_pca,
    optim_pval = optim_pval
  )
  
  
  x <- data$X
  y <- data$Y
  U <- data[, grep("U.", names(data))]
  Z <- data[, grep("Z.", names(data))]
  
  results <- lapply(methods, function(meth){
    meth(x = x, y = y, proxy = U, rank = 20)
  })
  
  ### save results:
  
  for (nm in names(results)){
    dir.create(file.path(res_dir, nm), recursive = TRUE, showWarnings = FALSE)
    write.csv(results[[nm]], file = file.path(res_dir, nm, name), row.names = FALSE)
  }
  
  print(args[c("noise", "dist")])
  print(sapply(results, function(res){
    sapply(evaluations, function(eval) eval(res, Z, args))
  }))
}