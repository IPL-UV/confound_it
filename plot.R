allfiles <- list.files("results/", pattern = "*.csv", recursive = FALSE, full.names = TRUE)


allres <- lapply(allfiles, read.csv)

DD <- reshape2::melt(allres, id.vars = names(allres[[1]]))
DD$ab_c_inv <- 1 / DD$ab_c

library(ggplot2)

ggplot(DD) + stat_smooth(aes(y = value, x = abs(ab_c), group = method, colour = method)) + 
  facet_grid(rows = vars(dist), cols = vars(stats))
