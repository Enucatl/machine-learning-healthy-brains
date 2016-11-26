#!/usr/bin/env Rscript

library(argparse)
library(data.table)
library(ggplot2)
library(RcppCNPy)

parser = ArgumentParser()
parser$add_argument("file")
parser$add_argument("health")
parser$add_argument("output")
args = parser$parse_args()

dt = npyLoad(args$file)
ht = fread(args$health)
setnames(ht, "V1", "health")
ds = dist(dt, method="maximum")
h = hclust(ds, method="complete")
print(dt)

clustering = as.dendrogram(h)
colors = c("red", "blue")
col.labels = function(x) {
    if (is.leaf(x)) {
        a = attributes(x)
        color = colors[ht[a$label, health] + 1]
        attr(x, "nodePar") = c(a$nodePar, lab.col=color)
    }
    x
}

dend = dendrapply(clustering, col.labels)

predictions = cutree(h, 2)
ht[, prediction := predictions]
output.table = data.table(group=predictions)
print(output.table)
write.csv(output.table, args$output, row.names=FALSE)
print(ht)


width = 21
factor = 0.618
height = width * factor
output = dev.new(width=width, height=height)
plot(dend)

#invisible(readLines("stdin", n=1))
