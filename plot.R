library(ggplot2)
source("evaluations.R")


allfiles <- list.files("results", pattern = "evals_.*.csv", recursive = TRUE, full.names = TRUE)


allres <- lapply(allfiles, read.csv)

nm <- c(names(evaluations), "pvals_adj.px", "pvals_adj.py",
        "pvals_naive.px", "pvals_naive.py")
DD <- reshape2::melt(allres, measure.vars = nm[c(1:6)],
                     variable.name = "stats")

#DD <- DD[DD$size == 500, ] ## fix size  or proxy

ggplot(DD[DD$stats %in% c("cor"), ],
       mapping =  aes(y = value, x = abs(conf_x + conf_y), group = method, colour = method)) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  geom_point(alpha = 0.8) +
  #scale_x_log10() + 
  #scale_y_continuous(limits = c(0,1)) +
  #scale_y_log10() + 
  facet_grid(cols = vars(size, proxy),
             rows = vars(dist, noise,), scales = "free", labeller = label_context) + 
  ylab("cor") +
  theme_bw()

ggplot(DD[DD$stats %in% c("cor"), ],
       mapping =  aes(y = value, x = abs(ab_c), group = method, colour = method)) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  geom_point(alpha = 0.8) +
  scale_x_log10() + 
  #scale_y_continuous(limits = c(0,1)) +
  #scale_y_log10() + 
  facet_grid(cols = vars(size, proxy),
             rows = vars(dist, noise,), scales = "free", labeller = label_context) + 
  ylab("cor") +
  theme_bw()

ggplot(DD[DD$stats %in% c("ae_cc_vs_ols"), ]) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  geom_abline(intercept = log(1, base = 10), slope = 0, color = "black") +
  #geom_rect(data=data.frame(xmin = 0, xmax = Inf, ymin = 1, ymax = Inf),
  #          aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = "white", color = NULL), show.legend = FALSE) + 
  geom_point(aes(y = value, x = abs(conf_x + conf_y), group = method, colour = method), alpha = 0.8) +
  #scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  scale_y_log10() + 
  facet_grid(cols = vars(proxy, size),
             rows = vars(dist, noise), scales = "free", labeller = label_context) + 
  ylab("ae_cc_vs_ols")  + 
  theme_bw()


ggplot(DD[DD$stats %in% c("cor"), ],
       mapping =  aes(y = pvals_naive.px + pvals_naive.py, x = abs(proxy), group = method, colour = method)) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  geom_point(alpha = 0.8) +
  scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  scale_y_log10() + 
  facet_grid(cols = vars(),
             rows = vars(dist, noise), scales = "free", labeller = label_context) + 
  theme_bw()



DDmean <- do.call(data.frame, aggregate(DD[, c("value", "pvals_naive.px", "pvals_naive.py")], by = list(size = DD$size, method = DD$method,
                                                            proxy = DD$proxy,
                                  noise = DD$noise, dist = DD$dist, stats = DD$stats),
                    FUN = function(x) return(c(median = unname(median(x, na.rm = TRUE)), min = unname(quantile(x, probs = 0.025, na.rm = TRUE)),
                                               max = unname(quantile(x, probs = 0.975, na.rm = TRUE)))), simplify = TRUE))


ggplot(DDmean[DDmean$stats  %in% c("ae_cc_vs_ols"), ],
       mapping =  aes(y = value.median, x = proxy, colour = method, fill = method)) + 
  geom_line() +  geom_abline(slope = 0, intercept = log(1, 10)) +
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()

ggplot(DDmean[DDmean$stats  %in% c("ae_cc_vs_proxy"), ],
       mapping =  aes(y = value.median, x = pvals_naive.px.median + pvals_naive.py.median, colour = method, fill = method)) + 
  geom_point() + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()


ggplot(DDmean[DDmean$stats  %in% c("cor"), ],
       mapping =  aes(y = value.median, x = proxy, colour = method, fill = method)) + 
  geom_line() + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + #scale_y_log10() + 
  scale_y_continuous(limits = c(0,1)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()


ggplot(DDmean[DDmean$stats  %in% c("se_cc"), ],
       mapping =  aes(y = value.median, x = proxy, colour = method, fill = method)) + 
  geom_line() + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,1)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()

