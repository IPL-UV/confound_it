library(ggplot2)
source("evaluations.R")


allfiles <- list.files("results/", pattern = "evals_.*.csv", recursive = TRUE, full.names = TRUE)


allres <- lapply(allfiles, read.csv)

DD <- reshape2::melt(allres, measure.vars = names(evaluations),
                     variable.name = "stats")

ggplot(DD[DD$stats %in% c("cor", "ae_cc_norm"), ],
       mapping =  aes(y = value, x = abs(causal_coeff), group = method, colour = method)) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  geom_point(alpha = 0.8) +
  #scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  #scale_y_log10() + 
  facet_grid(cols = vars(dist, noise, size),
             rows = vars(stats), scales = "free", labeller = label_context)

ggplot(DD[DD$stats %in% c("ae_cc_norm"), ],
       mapping =  aes(y = value, x = abs(ab_c), group = method, colour = method)) + 
  stat_smooth(method =  "loess", se = FALSE) + geom_point(alpha = 0.3) +
  #scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  scale_y_log10() + 
  facet_grid(cols = vars(dist, noise, size),
             rows = vars(stats), scales = "free", labeller = label_context)


DDmean <- do.call(data.frame, aggregate(DD$value, by = list(size = DD$size, method = DD$method,
                                  noise = DD$noise, dist = DD$dist, stats = DD$stats),
                    FUN = function(x) return(c(mean = mean(x), min = unname(quantile(x, probs = 0.025)),
                                               max = unname(quantile(x, probs = 0.975)))), simplify = TRUE))

ggplot(DDmean[DDmean$stats  %in% c("cor", "se_cc", "ae_cc_norm"), ],
       mapping =  aes(y = x.mean, x = size, colour = method, fill = method)) + 
  geom_line() + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context)

