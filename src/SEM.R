library(lavaan)
library(semPlot)
data <- read.csv("data/raw/sem_synthetic/sem_synthetic.csv")

model <- ' 
  Extraversion =~ ext_it1 + ext_it2 + ext_it3 + ext_it4
  Neuroticism =~ neu_it1 + neu_it2 + neu_it3 + neu_it4
  Conscientiousness =~ con_it1 + con_it2 + con_it3 + con_it4
  
  Extraversion ~ Neuroticism
  Extraversion ~~ Conscientiousness 
'

fit <- sem(model, data)
summary(fit)

semPaths(fit, "std", layout="tree", edge.label.cex = 1.2)

model_indices <- fitmeasures(fit, fit.measures = "all")
model_indices

modindices(fit)
