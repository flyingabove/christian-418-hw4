library(keras)
library(magrittr)

###Cleaning###
sentiment_df<-fread("data/sentiment_raw.csv")
sentiment_df$ascii<-iconv(sentiment_df$JustText, "latin1", "ASCII", sub="")
sentiment_df$no_repeats<-gsub("([[:alpha:]])\\1{2,}", "\\1", sentiment_df$ascii)
sentiment_df$no_repeats<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",perl=TRUE,sentiment_df$no_repeats)
sentiment_df<-sentiment_df[sapply(gregexpr("\\W+", sentiment_df$no_repeats), length) >1,]

sentiment_df_small<-sentiment_df[1:10000,]
writeLines(sentiment_df_small$no_repeats,"smalltest.txt")

factor_list<-factor(readLines("wordList.txt"))
text_raw_list<-strsplit(sentiment_df_small$no_repeats,split = " ")

get_training<-function(string_in,levels){
  y2<-factor(string_in,levels = levels); 
  result<-unclass(y2) %>% as.numeric 
  if(NA %in% result){
    print(string_in)
    print(result)
  }
  result
}

nn_predictors<-lapply(text_raw_list,get_training,levels = factor_list)

training_index<-sample(1:length(nn_predictors),length(nn_predictors)*.75)
x_train<- nn_predictors[training_index]
x_test <- nn_predictors[-training_index]

y_train<- sentiment_df_small$Sentiment[training_index]
y_test <- sentiment_df_small$Sentiment[-training_index]

### Test and Train ###

max_features <- length(factor_list)
maxlen <- 80  # cut texts after this number of words 
batch_size <- 32

print('Loading data...\n')
print(length(x_train), 'train sequences\n')
print(length(x_test), 'test sequences\n')

print('Pad sequences (samples x time)\n')
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
print('x_train shape:', dim(x_train), '\n')
print('x_test shape:', dim(x_test), '\n')

print('Build model...\n')
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# try using different optimizers and different optimizer configs
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

print('Train...\n')
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 1,
  validation_data = list(x_test, y_test)
)
scores <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size
)

cat('Test score:', scores[[1]])
cat('Test accuracy', scores[[2]])
