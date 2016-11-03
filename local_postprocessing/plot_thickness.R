#!/usr/bin/env Rscript

library(argparse)
library(data.table)
library(ggplot2)

parser = ArgumentParser()
parser$add_argument("file", default="../data/cortical_thickness.csv")
args = parser$parse_args()

dt = fread(args$file)
print(dt)

plot = ggplot(dt) +
    geom_histogram(
        aes(x=thickness, y=..density.., fill=factor(health)),
        alpha=0.4, binwidth=1, position="identity") +
    xlim(0, 20) +
    scale_fill_discrete(
        name="",
        breaks=c(0, 1),
        labels=c("unhealthy", "healthy"))
print(plot)

width = 7
factor = 0.618
height = width * factor
ggsave("cortical_thickness.png", plot, width=width, height=height, dpi=300)
invisible(readLines("stdin", n=1))
