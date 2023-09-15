allfiles <- list.files("results/", pattern = "*.csv", recursive = FALSE, full.names = TRUE)


allres <- lapply(allfiles, read.csv)

DD <- reshape2::melt(allres, id.vars = names(allres[[1]]))


library(ggplot2)

ggplot(DD) + geom_line(aes(y = value, x = abs(ab_c), group = method, colour = method)) + 
  facet_grid(rows = vars(dist, noise), cols = vars(stats))
