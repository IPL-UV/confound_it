  library(ggplot2)
  source("evaluations.R")
  
  ##### load all results and reshape
  allfiles <- list.files("evaluations", pattern = "evals_.*.csv", recursive = TRUE, full.names = TRUE)
  
  allres <- lapply(allfiles, read.csv)
  
  nm <- names(evaluations)
  DD <- reshape2::melt(allres, measure.vars = nm[c(1:9)],
                       variable.name = "stats")
  #DD <- DD[DD$size > 10,]
  DD$pvals <- DD$pvals_naive.px + DD$pvals_naive.py
  DD$method <- as.factor(DD$method)
  levels(DD$method) <- list(oracle = "oracle",
                            "ICA-PCF" = "ica_sel_naive", 
                            "PCA-PCF" = "pca_sel_naive",
                            "PLS-PCF" = "pls_sel_naive",
                            "GD-PCF" = "gd_pcf")
  
  levels(DD$stats) <- list("AbsCor" = "cor",
                           "ae_cc_vs_pearlpca" = "ae_cc_vs_pearlpca",
                           "ae_cc_vs_pearlfirst" = "ae_cc_vs_pearlfirst",
                           "ae_cc_vs_ols" = "ae_cc_vs_ols", 
                           "ae_cc_vs_proxy" = "ae_cc_vs_proxy",
                           "ae_cc_vs_proxysmall" = "ae_cc_vs_proxysmall",
                           "AE"  = "ae_cc",   
                           "AER" = "ae_cc_vs_pca",
                           "are_cc" = "are_cc" , 
                           "se_cc" = "se_cc")
  
  
  ##### compute  median values
  DD1 <-  DD
  
  DDM <- do.call(data.frame, aggregate(DD1[, c("value", "pvals")],
                                       by = DD1[,c("size", "method", "proxy",
                                                   "noise", "dist", "stats")],
                                       FUN = function(x){
                                         return(c(median = unname(median(x, na.rm = TRUE)),
                                                  min = unname(quantile(x, probs = 0.25, na.rm = TRUE)),
                                                  max = unname(quantile(x, probs = 0.75, na.rm = TRUE))))},
                                       simplify = TRUE))
  
  sizes <- sort(unique(DD$size))
  
  
  #DDM[DDM$stats == "AE ratio" & DDM$value.median > 3,] <- NA
  ggplot(data = DDM[DDM$stats %in% c("AbsCor", "AE", "AER") & DDM$dist %in% c("gaussian", "exponential"),]) + 
    #geom_hline(mapping = aes(yintercept = y, color = method, linetype = method), data = data.frame(stats = c("AbsCor", "AE ratio"), y = 1, method = c("oracle", NA)), linewidth = 1 ) +
    geom_line(mapping = aes(x = size, y = value.median, colour = method, linetype = method), 
              linewidth = 1) + 
    #geom_ribbon(aes(x = size, ymin = value.min, ymax = value.max, colour = method), alpha = 0.3) + 
    scale_x_log10(breaks = sizes, labels = sizes) + scale_y_log10() + 
    #coord_cartesian(ylim = c(0,1)) +
    #scale_y_continuous(limits = c(0,100)) +
    ylab("")  + 
    scale_linetype_manual(values = c("oracle" = "dotdash",
                                     "ICA-PCF" = "solid",
                                     "PCA-PCF" = "dotted",
                                     "PLS-PCF" = "dashed",
                                     "GD-PCF"= "twodash")) + 
    scale_color_manual(values = c("oracle" = "black",
                                  "ICA-PCF" = "blue",
                                  "PCA-PCF" = "red",
                                  "PLS-PCF" = "purple",
                                  "GD-PCF" = "green"
                                  )) + 
    facet_grid(cols = vars(dist), rows = vars(stats), scales = "free") + theme_bw() + 
    theme(legend.position = "right", legend.title = element_blank(),
          plot.margin = unit(c(0,0,0,0), units = "in"),
          axis.title.x = element_text(margin = unit(c(0,0,0,0), "in")),
          legend.box.spacing = unit(0, "cm"), axis.text.x = element_text(angle = 45))
  ggsave("plot_all.pdf", width = 5.5, height = 2.5, units = "in")
    
  
  #DDM[DDM$stats == "AE ratio" & DDM$value.median > 3,] <- NA
  ggplot(data = DDM[DDM$stats %in% c("AbsCor", "AE", "AER") & DDM$dist %in% c("uniform", "gamma"),]) + 
    #geom_hline(mapping = aes(yintercept = y, color = method, linetype = method), data = data.frame(stats = c("AbsCor", "AE ratio"), y = 1, method = c("oracle", NA)), linewidth = 1 ) +
    geom_line(mapping = aes(x = size, y = value.median, colour = method, linetype = method), 
              linewidth = 1) + 
    #geom_ribbon(aes(x = size, ymin = value.min, ymax = value.max, colour = method), alpha = 0.3) + 
    scale_x_log10(breaks = sizes, labels = sizes) + scale_y_log10() + 
    #coord_cartesian(ylim = c(0,1)) +
    #scale_y_continuous(limits = c(0,100)) +
    ylab("")  + 
    scale_linetype_manual(values = c("oracle" = "dotdash",
                                     "ICA-PCF" = "solid",
                                     "PCA-PCF" = "dotted",
                                     "PLS-PCF" = "dashed",
                                     "GD-PCF" = "twodash",
                                     "optim_res" = "solid")) + 
    scale_color_manual(values = c("oracle" = "black",
                                  "ICA-PCF" = "blue",
                                  "PCA-PCF" = "red",
                                  "PLS-PCF" = "purple",
                                  "GD-PCF" = "green",
                                  "optim_res" = "magenta")) + 
    facet_grid(cols = vars(dist), rows = vars(stats), scales = "free") + theme_bw() + 
    theme(legend.position = "right", legend.title = element_blank(),
          plot.margin = unit(c(0,0,0,0), units = "in"),
          axis.title.x = element_text(margin = unit(c(0,0,0,0), "in")),
          legend.box.spacing = unit(0, "cm"), axis.text.x = element_text(angle = 45))
  ggsave("plot_supp.pdf", width = 5.5, height = 2.5, units = "in")
  
