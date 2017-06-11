
library(h2o)
h2o.init(nthreads = -1,min_mem_size = "5g")

df <- h2o.importFile("/home/christian/Documents/christian-418-hw4/data/sentiment_dt_small.csv")

write.csv(test[1:15000,],"data/sentiment_dt_small.csv",row.names = FALSE)

model<-load_model_hdf5(filepath="data/nn_small_model",custom = NULL)
