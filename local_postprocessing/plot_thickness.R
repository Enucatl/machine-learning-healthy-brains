#!/usr/bin/env Rscript

library(argparse)
library(data.table)
library(ggplot2)

parser = ArgumentParser()
parser$add_argument("file", default="../data/cortical_thickness.csv")
args = parser$parse_args()

dt = fread(args$file)[thickness <= 20]
print(dt)
cumulative = dt[,
    .(
        cumulative=ecdf(thickness)(seq(0, 20, 1)),
        thickness=seq(0, 20, 1)
        )
        , by=list(id, health)]
setkey(cumulative, id)
print(cumulative)
cumulative.wide = dcast(cumulative, id + health ~ thickness, value.var="cumulative")
print(cumulative.wide)
ids = as.character(cumulative.wide[, id])
healths = cumulative.wide[, health]
id.table = cumulative.wide[, .(id, health)]
cumulative.wide[, health := NULL]
cumulative.wide[, id := NULL]
print(cumulative.wide)
print(id.table)
cumulative.matrix = as.matrix(cumulative.wide)

ds = dist(cumulative.matrix, method="maximum")
h = hclust(ds, method="complete")

clustering = as.dendrogram(h)
colors = c("red", "blue")
col.labels = function(x) {
    if (is.leaf(x)) {
        a = attributes(x)
        color = colors[id.table[id == a$label - 1, health] + 1]
        attr(x, "nodePar") = c(a$nodePar, lab.col=color)
    }
    x
}

dend = dendrapply(clustering, col.labels)

predictions = 1 - (cutree(h, 2) - 1)
id.table[, prediction := predictions]
score.table = id.table[, .(
    tp=(health == 0 & prediction == 0),
    p=(health == 0),
    tn=(health == 1 & prediction == 1),
    n=(health == 1))]
sensitivity = score.table[, sum(tp) / sum(p)]
specificity = score.table[, sum(tn) / sum(n)]
prevalence = score.table[, sum(p) / (sum(p) + sum(n))]
print("sensitivity:")
print(sensitivity)
print("specificity:")
print(specificity)
print("prevalence:")
print(prevalence)



calc.score = function(prediction) {
    p.sick.given.positive = (sensitivity * prevalence) / (sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
    p.healthy.given.negative = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)
    return((1 - prediction) * (1 - p.sick.given.positive) + prediction * p.healthy.given.negative)
}

id.table[, score := calc.score(prediction)]
print(id.table)

binary.log.loss = id.table[, sum(health * log(score) + (1 - health) * log(1 - score)) / (-.N)]
print(binary.log.loss)


width = 21
factor = 0.618
height = width * factor
output = dev.new(width=width, height=height)
plot(dend)

invisible(readLines("stdin", n=1))
