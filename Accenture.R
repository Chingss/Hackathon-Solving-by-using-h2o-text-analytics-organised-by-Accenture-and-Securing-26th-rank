# Loading The Required Package,s

library(SentimentAnalysis)
library(Hmisc)
library(missForest)

# Choosing The Working Directory

setwd(choose.dir())

# Loading the train file 

train <- read.csv("train.csv")

train$comment <- as.character(train$comment)
train$parent_comment <- as.character(train$parent_comment)


train1 <- train[1:15000,]
train2 <- train[15001:30000,]
train3 <- train[30001:45000,]

?analyzeSentiment

##################
################## 

## Using sentimentAnalysis() Package to convert the text data into numerric data based on their 
## scoring on Positivity and Negativity

sent_1_comm <- analyzeSentiment(train1$comment)
sent_2_comm <- analyzeSentiment(train2$comment)
sent_3_comm <- analyzeSentiment(train3$comment)

sentiment_comment <- as.data.frame(rbind(sent_1_comm,sent_2_comm,sent_3_comm))
rm(sent_1_comm,sent_2_comm,sent_3_comm)


sent_1_pare <- analyzeSentiment(train1$parent_comment)
sent_2_pare <- analyzeSentiment(train2$parent_comment)
sent_3_pare <- analyzeSentiment(train3$parent_comment)


sentiment_parent <- as.data.frame(rbind(sent_1_pare,sent_2_pare,sent_3_pare))

rm(sent_1_pare,sent_2_pare,sent_3_pare)

rm(train1,train2,train3)

train$comment <- NULL
train$parent_comment <- NULL

names(sentiment_parent)

colnames(sentiment_parent) <- c( "WordCount_P","SentimentGI_P","NegativityGI_P","PositivityGI_P",      
                                 "SentimentHE_P"  ,      "NegativityHE_P"    ,   "PositivityHE_P" ,      "SentimentLM_P","NegativityLM_P","PositivityLM_P","RatioUncertaintyLM_P","SentimentQDAP_P",     
                                 "NegativityQDAP_P" ,    "PositivityQDAP_P")




tr <- as.data.frame(cbind(train,sentiment_comment,sentiment_parent))

# removing Observations containg Missing Values

describe(tr)
tr <- tr[complete.cases(tr),]

rm(sentiment_comment,sentiment_parent)

saveRDS(tr,"tr.rds")
tr <- readRDS("tr.rds")

# Test 
test <- read.csv("test.csv")
test$comment <- as.character(test$comment)
test$parent_comment <- as.character(test$parent_comment)

test1 <- test[1:10000,]
test2 <- test[10001:20000,]
test3 <- test[20001:30000,]

sent_1_comm_t <- analyzeSentiment(test1$comment)
sent_2_comm_t <- analyzeSentiment(test2$comment)
sent_3_comm_t <- analyzeSentiment(test3$comment)

sentiment_comment_t <- as.data.frame(rbind(sent_1_comm_t,sent_2_comm_t,sent_3_comm_t))

rm(sent_1_comm_t,sent_2_comm_t,sent_3_comm_t)

sent_1_pare_t <- analyzeSentiment(test1$parent_comment)
sent_2_pare_t <- analyzeSentiment(test2$parent_comment)
sent_3_pare_t <- analyzeSentiment(test3$parent_comment)

sentiment_parent_t <- as.data.frame(rbind(sent_1_pare_t,sent_2_pare_t,sent_3_pare_t))

rm(sent_1_pare_t,sent_2_pare_t,sent_3_pare_t)
rm(test1,test2,test3)

test$comment <- NULL
test$parent_comment <- NULL

colnames(sentiment_parent_t) <- c( "WordCount_P","SentimentGI_P","NegativityGI_P","PositivityGI_P",      
                                   "SentimentHE_P"  ,      "NegativityHE_P"    ,   "PositivityHE_P" ,      "SentimentLM_P","NegativityLM_P","PositivityLM_P","RatioUncertaintyLM_P","SentimentQDAP_P",     
                                   "NegativityQDAP_P" ,    "PositivityQDAP_P")
te <- as.data.frame(cbind(test,sentiment_comment_t,sentiment_parent_t))
rm(sentiment_comment_t,sentiment_parent_t)

# treating missing values generated

# test 

#missForest
te$UID <- as.character(te$UID)

te.imp <- missForest(te[,3:30])
str(te.imp)
describe(te.imp$ximp)

te[,3:30] <- NULL

te <- as.data.frame(cbind(te,te.imp$ximp))
rm(te.imp)
describe(te)

saveRDS(te,"te.rds")
te <- readRDS("te.rds")


write.csv(te,"te.csv",row.names = F)
write.csv(tr,"tr.csv",row.names = F)


library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()   # Initializing H2O Environment

tr_new_h20 <- as.h2o(tr) # converting train file into h2o data frame
te_fin_h20 <- as.h2o(te) # converting test file into h2o data frame

y.dep <- 3
x.dep <- c(4:31)

# Generate a random grid of models and stack them together

# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     x = x.dep,
                     y = y.dep,
                     training_frame = tr_new_h20,
                     model_id = "gbm_grid_regression",
                     ntrees = 10,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
summary(gbm_grid)

# Train a stacked ensemble using the GBM grid <- (24.34379)
ensemble <- h2o.stackedEnsemble(x = x.dep,
                                y = y.dep,
                                training_frame = tr_new_h20,
                                model_id = "ensemble_gbm_grid_regression",
                                base_models = gbm_grid@model_ids)

h2o.performance(ensemble)

predict.ense.grid <- as.data.frame(h2o.predict(ensemble, te_fin_h20[,3:30]))
sub_ense.g <- as.data.frame(cbind(UID = te$UID,score = predict.ense.grid$predict))
write.csv(sub_ense.g, file = "sub_ense_g.csv", row.names = F)


h2o.shutdown(prompt=FALSE)


