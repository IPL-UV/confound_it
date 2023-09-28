library(ggplot2)
source("evaluations.R")

##### load all results and reshape
allfiles <- list.files("evaluations/", pattern = "evals_.*.csv", recursive = TRUE, full.names = TRUE)

allres <- lapply(allfiles, read.csv)

nm <- c(names(evaluations), "pvals_adj.px", "pvals_adj.py",
        "pvals_naive.px", "pvals_naive.py")
DD <- reshape2::melt(allres, measure.vars = nm[c(1:8)],
                     variable.name = "stats")

#methods <- c("ica_sel_naive", "oracle", "pca_sel", "pls_sel_naive")
DD$pvals_naive <- DD$pvals_naive.px + DD$pvals_naive.py
#DD <- DD[DD$method %in% methods,]
### pretty names
#levels(DD$stats) <- list("cor" = "cor",
#                         "ae_cc_vs_ols" = "ae_cc_vs_ols",
#                         "ae_cc_vs_proxy" = "ae_cc_vs_proxy",
#                         "ae_cc" = "ae_cc", 
#                         "are_cc" = "are_cc", 
#                         "se_cc" = "se_cc")


############ plot all individual experiments

ggplot(DD[DD$stats %in% c("ae_cc_vs_pearlfirst"), ],
       mapping =  aes(y = value, x = abs(pvals_naive), group = method, colour = method)) + 
  geom_point(alpha = 0.8) +
  #stat_smooth(method =  "loess", se = FALSE) + 
  scale_x_log10() + 
  #scale_y_continuous(limits = c(0,1)) +
  #scale_y_log10() + 
  facet_grid(cols = vars(size, proxy),
             rows = vars(dist, noise,), scales = "free", labeller = label_context) + 
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
  #geom_rect(data=data.frame(xmin = 0, xmax = Inf, ymin = 1, ymax = Inf),
  #          aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = "white", color = NULL), show.legend = FALSE) + 
  geom_point(aes(y = value, x = abs(ab_c), group = method, colour = method), alpha = 0.8) +
  #scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  geom_abline(intercept = log(1, base = 10), slope = 0, color = "black") +
  scale_y_log10() + 
  facet_grid(cols = vars(proxy, size),
             rows = vars(dist, noise), scales = "free", labeller = label_context) + 
  ylab("ae_cc_vs_ols")  + 
  theme_bw()

ggplot(DD[DD$stats %in% c("ae_cc_vs_proxysmall"), ]) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  #geom_rect(data=data.frame(xmin = 0, xmax = Inf, ymin = 1, ymax = Inf),
  #          aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = "white", color = NULL), show.legend = FALSE) + 
  geom_point(aes(y = value, x = abs(conf_x + conf_y), group = method, colour = method), alpha = 0.8) +
  #scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  geom_abline(intercept = log(1, base = 10), slope = 0, color = "black") +
  scale_y_log10() + 
  facet_grid(cols = vars(proxy, size),
             rows = vars(dist, noise), scales = "free", labeller = label_context) + 
  ylab("ae_cc_vs_proxysmall")  + 
  theme_bw()

ggplot(DD[DD$stats %in% c("ae_cc_vs_pca"), ]) + 
  #stat_smooth(method =  "loess", se = FALSE) + 
  #geom_rect(data=data.frame(xmin = 0, xmax = Inf, ymin = 1, ymax = Inf),
  #          aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = "white", color = NULL), show.legend = FALSE) + 
  geom_point(aes(y = value, x = abs(conf_x + conf_y), group = method, colour = method), alpha = 0.8) +
  #scale_x_log10() + #scale_y_continuous(limits = c(0,1)) +
  geom_abline(intercept = log(1, base = 10), slope = 0, color = "black") +
  scale_y_log10() + 
  facet_grid(cols = vars(proxy, size),
             rows = vars(dist, noise), scales = "free", labeller = label_context) + 
  ylab("ae_cc_vs_pca")  + 
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



###############  median values
DD1 <-  DD
DDM <- do.call(data.frame, aggregate(DD1[, c("value", "pvals_naive.px", "pvals_naive.py")],
                                        by = DD1[,c("size", "method", "proxy",
                                                    "noise", "dist", "stats")],
                                        FUN = function(x){
                                          return(c(median = unname(median(x, na.rm = TRUE)),
                                                   min = unname(quantile(x, probs = 0.25, na.rm = TRUE)),
                                                   max = unname(quantile(x, probs = 0.75, na.rm = TRUE))))},
                                        simplify = TRUE))

sizes <- sort(unique(DD$size))
DD$groups <- paste0(DD$method, DD$size)

### correlation plot
ggplot(DDM[DDM$stats  %in% c("cor") & !(DDM$method == "oracle") , ],
       mapping =  aes(y = value.median, x = size, colour = method, fill = method, linetype = method)) + 
  geom_line() +  geom_abline(slope = 0, intercept = log(1, 10), linetype = 4) +
  #geom_point(data = DD[DD$stats  %in% c("cor", "ae_cc_vs_pca"),], aes(y = value, x = size, colour = method, fill = method)) +
  #geom_ribbon(aes(x = size, ymin = value.min, ymax = value.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10(breaks = sizes, labels = sizes) + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("cor")  + 
  facet_grid(cols = vars(dist), rows = vars(), scales = "free",
             labeller = label_context) + theme_bw() + 
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.box.spacing = unit(0, "cm"), axis.text.x = element_text(angle = 45))
ggsave("plot_correlation.pdf", width = 5, height = 2, units = "in")


ggplot(DDM[DDM$stats  %in% c("ae_cc_vs_pca"), ],
       mapping =  aes(y = value.median, x = size, colour = method, fill = method, linetype = method)) + 
   geom_abline(slope = 0, intercept = log(1, 10), linetype = 4) +
  geom_boxplot(data = DD[DD$stats  %in% c("ae_cc_vs_pca"),], aes(y = value, x = size, colour = method, fill = method, group = groups)) +
  #geom_line() + 
  #geom_ribbon(aes(x = size, ymin = value.min, ymax = value.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10(breaks = sizes, labels = sizes) + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("")  + 
  facet_grid(cols = vars(dist), rows = vars(), scales = "free",
             labeller = label_context) + theme_bw() + 
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.box.spacing = unit(0, "cm"), axis.text.x = element_text(angle = 45))
ggsave("plot_ae_cc_vs_pca.pdf", width = 5, height = 2, units = "in")
## 


ggplot(DDM[DDM$stats  %in% c("ae_cc"), ],
       mapping =  aes(y = value.median, x = size, colour = method, fill = method, linetype = method)) + 
  geom_line() +  #geom_abline(slope = 0, intercept = log(1, 10), linetype = 4) +
  #geom_point(data = DD[DD$stats  %in% c("cor", "ae_cc_vs_pca"),], aes(y = value, x = size, colour = method, fill = method)) +
  geom_ribbon(aes(x = size, ymin = value.min, ymax = value.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10(breaks = sizes, labels = sizes) + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("")  + 
  facet_grid(cols = vars(dist), rows = vars(), scales = "free",
             labeller = label_context) + theme_bw() + 
  theme(legend.position = "bottom", legend.title = element_blank(),
        legend.box.spacing = unit(0, "cm"), axis.text.x = element_text(angle = 45))
ggsave("plot_ae_cc.pdf", width = 5, height = 2, units = "in")

ggplot(DD[DD$stats  %in% c("ae_cc_vs_pca") , ],
       mapping =  aes(y = value, x = as.factor(size), colour = method, group = groups)) + 
  geom_boxplot() +  geom_abline(slope = 0, intercept = log(1, 10)) +
  #geom_point(data = DD[DD$stats  %in% c("cor", "ae_cc_vs_pca"),], aes(y = value, x = size, colour = method, fill = method)) +
  #scale_x_log10(breaks = sizes, labels = sizes) + 
  scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("") + 
  facet_grid(cols = vars(method), rows = vars(dist), scales = "fixed",
             labeller = label_context) + theme_bw()

ggplot(DDM[DDM$stats  %in% c("ae_cc_vs_proxy", "ae_cc_vs_proxysmall", "ae_cc_vs_pca"), ],
       mapping =  aes(y = value.median, x = proxy, colour = method, fill = method)) + 
  geom_line() + 
  geom_abline(slope = 0, intercept = 0) + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,100)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()


ggplot(DDM[DDM$stats  %in% c("cor"), ],
       mapping =  aes(y = value.median, x = proxy, colour = method, fill = method)) + 
  geom_line() + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + #scale_y_log10() + 
  scale_y_continuous(limits = c(0,1)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()


ggplot(DDM[DDM$stats  %in% c("se_cc"), ],
       mapping =  aes(y = value.median, x = proxy, colour = method, fill = method)) + 
  geom_line() + 
  #geom_ribbon(aes(x = size, ymin = x.min, ymax = x.max, colour = NULL), alpha = 0.3) + 
  scale_x_log10() + scale_y_log10() + 
  #scale_y_continuous(limits = c(0,1)) +
  ylab("") + 
  facet_grid(cols = vars(dist, noise), rows = vars(stats), scales = "free",
             labeller = label_context) + theme_bw()

