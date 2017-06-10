##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)
library(gbm)
library(randomForest)
library(xgboost)

###############
##### GBM #####
###############

###GBM- H2o###
h2o.init(nthreads = -1)
df <- h2o.importFile("data/sentiment_df-gbm.csv")

splits <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  ratios = c(.1,.05),   ##  create splits 
  seed=1234)    

train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")

###GBM Base Model###

gbm_base<-h2o.gbm(y = "Sentiment", training_frame = train)
gbm_base
h2o.auc(h2o.performance(gbm_base, newdata = valid))
plot(h2o.performance(gbm_base, newdata = valid_2),col = "blue",main = "True Positives vs False Positives GBM")
