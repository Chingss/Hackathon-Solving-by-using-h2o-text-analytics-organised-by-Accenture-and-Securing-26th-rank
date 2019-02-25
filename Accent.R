library(SentimentAnalysis)


setwd("C:/Users/DELL PC/Desktop/Hackathons/accenture/35ed93d0-2-dataset/dataset")


train <- read.csv("train.csv")


train$comment <- as.character(train$comment)
train$parent_comment <- as.character(train$parent_comment)


train1 <- train[1:15000,]
train2 <- train[15001:30000,]
train3 <- train[30001:45000,]

?analyzeSentiment

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


rm(sentiment_comment,sentiment_parent)


library(Hmisc)
describe(tr)
tr <- tr[complete.cases(tr),]

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


saveRDS(tr,"tr.rds")


tr <- readRDS("tr.rds")



# treating missing values generated

# test 

#missForest
install.packages("missForest")
library(missForest)
te$UID <- as.character(te$UID)

iris.imp <- missForest(te[,3:30])
str(iris.imp)
describe(iris.imp$ximp)

te[,3:30] <- NULL

te <- as.data.frame(cbind(te,iris.imp$ximp))
rm(iris.imp)
describe(te)

saveRDS(te,"te.rds")
te <- readRDS("te.rds")
write.csv(te,"te.csv",row.names = F)
write.csv(tr,"tr.csv",row.names = F)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(tr))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(tr)), size = smp_size)

c.tr <- tr[train_ind, ]
c.te <- tr[-train_ind, ]

c.te_label <- c.te$score
c.te$score <- NULL

install.packages("h2o")
library(h2o)


localH2O <- h2o.init(nthreads = -1)

h2o.init()
tr_new_h20 <- as.h2o(tr)
te.h2o <- as.h2o(c.te)
tr.h2o <- as.h2o(c.tr)
c.te.h2o_labe <- as.h2o(c.te_label)
te_fin_h20 <- as.h2o(te)

y.dep <- 3
x.dep <- c(4:31)

#Random Forest
system.time(
  rforest.model <- h2o.randomForest(y=y.dep, x=x.dep, training_frame = tr_new_h20, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
)

h2o.performance(rforest.model)

#making predictions on test data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, te_fin_h2o)))
predict.rforest <- as.data.frame(h2o.predict(rforest.model, te_fin_h20[,3:30]))

sub_rf <- as.data.frame(cbind(UID = te$UID,score = predict.rforest$predict))
write.csv(sub_rf, file = "sub_rf.csv", row.names = F)

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}

RMSE(predict.rforest$predict,c.te_label)


#GBM
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.dep,nfolds = 3, training_frame = tr_new_h20, ntrees = 1000, max_depth = 4, learn_rate = 0.1, seed = 1122)
)
?h2o.gbm
h2o.performance(gbm.model)

summary(gbm.model)

system.time(predict.gbm <- as.data.frame(h2o.predict(gbm.model, te_fin_h20[,3:30])))

predict.gbm <- as.data.frame(h2o.predict(gbm.model, te_fin_h20[,3:30]))

RMSE(predict.gbm$predict,c.te_label)

sub_gbm <- as.data.frame(cbind(UID = te$UID,score = predict.gbm$predict))
write.csv(sub_gbm, file = "sub_gbm.csv", row.names = F)

#deep learning models
 system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.dep,
                                      training_frame = tr_new_h20,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier",
                                      seed = 1122
  )
)


 h2o.performance(dlearning.model)
 predict.dlearning <- as.data.frame(h2o.predict(dlearning.model, te_fin_h20[,3:30]))
 sub_dlearn <- as.data.frame(cbind(UID = te$UID,score = predict.dlearning$predict))
 write.csv(sub_dlearn, file = "sub_dlearn.csv", row.names = F)

# xgboost
# 
h2o.xgboost.available()
 
 system.time(
   xgboost.model <- h2o.xgboost.available(y=y.dep, x=x.dep,nfolds = 3, training_frame = tr_new_h20, ntrees = 1000, max_depth = 4, learn_rate = 0.3, seed = 1122)
 )
 
 ## AutoML <- Leading (24.34059)
 ?h2o.automl
 ?h2o.stackedEnsemble
 
 aml <- h2o.automl(x = x.dep, y = y.dep,
                   training_frame = tr_new_h20,
                   max_models = 20,
                   seed = 1)
 aml@leaderboard
 
 predict.aml <- as.data.frame(h2o.predict(aml, te_fin_h20[,3:30]))
 sub_aml <- as.data.frame(cbind(UID = te$UID,score = predict.aml$predict))
 write.csv(sub_aml, file = "sub_aml.csv", row.names = F)
 
 
 ### Stacked Ensemble
 ?h2o.gbm
 
 # Number of CV folds (to generate level-one data for stacking)
 nfolds <- 5
 # Train & Cross-validate a GBM
 my_gbm <- h2o.gbm(x = x.dep,
                   y = y.dep,
                   training_frame = tr_new_h20,
                   distribution = "AUTO",
                   ntrees = 10,
                   max_depth = 3,
                   min_rows = 2,
                   learn_rate = 0.2,
                   nfolds = nfolds,
                   fold_assignment = "Modulo",
                   keep_cross_validation_predictions = TRUE,
                   seed = 1)
 summary(my_gbm)
 
 predict.GBM2 <- as.data.frame(h2o.predict(my_gbm, te_fin_h20[,3:30]))
 sub_GBM2 <- as.data.frame(cbind(UID = te$UID,score = predict.GBM2$predict))
 write.csv(sub_GBM2, file = "sub_gbm2.csv", row.names = F)
 
 # Train & Cross-validate a RF
 my_rf <- h2o.randomForest(x = x.dep,
                           y = y.dep,
                           training_frame = tr_new_h20,
                           ntrees = 50,
                           nfolds = nfolds,
                           fold_assignment = "Modulo",
                           keep_cross_validation_predictions = TRUE,
                           seed = 1)
 summary(my_rf)
 predict.rf2 <- as.data.frame(h2o.predict(my_rf, te_fin_h20[,3:30]))
 sub_rf2 <- as.data.frame(cbind(UID = te$UID,score = predict.rf2$predict))
 write.csv(sub_rf2, file = "sub_rf2.csv", row.names = F)
 
 
 # Train a stacked ensemble using the GBM and RF above
 ensemble <- h2o.stackedEnsemble(x = x.dep,
                                 y = y.dep,
                                 training_frame = tr_new_h20,
                                 model_id = "my_ensemble_regression",
                                 base_models = list(my_gbm, my_rf))
 
 predict.ense <- as.data.frame(h2o.predict(ensemble, te_fin_h20[,3:30]))
 sub_ense <- as.data.frame(cbind(UID = te$UID,score = predict.ense$predict))
 write.csv(sub_ense, file = "sub_ense.csv", row.names = F)
 
 
 # 2. Generate a random grid of models and stack them together
 
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
 
 # Train a stacked ensemble using the GBM grid <- 24.34379
 ensemble <- h2o.stackedEnsemble(x = x.dep,
                                 y = y.dep,
                                 training_frame = tr_new_h20,
                                 model_id = "ensemble_gbm_grid_regression",
                                 base_models = gbm_grid@model_ids)
 
 predict.ense.grid <- as.data.frame(h2o.predict(ensemble, te_fin_h20[,3:30]))
 sub_ense.g <- as.data.frame(cbind(UID = te$UID,score = predict.ense.grid$predict))
 write.csv(sub_ense.g, file = "sub_ense_g.csv", row.names = F)
 
 
 h2o.shutdown(prompt=FALSE)

#SVR
rm(dlearning.model,gbm.model,rforest.model)
rm(te.h2o,te_fin_h20,tr_new_h20,tr.h2o)

install.packages("e1071")
library(e1071)

x <- subset(tr, select = -c(score,UID,date))
y <- tr$score
?h2o.xgboost
modelsvm <- svm(x,y)
summary(modelsvm)
te_te <- as.data.frame(te[3:30])
svm.predict <- as.data.frame(predict(modelsvm,te_te))
?svm
?svm.formula

d$y=as.numeric(d$y)
d$x=NULL


