#LOAN REQUEST CLASSIFICATION

#Libraries and options ####
library(tidyverse)
library(GGally)
library(DataExplorer)
library(gt)
library(ggedit)
library(tidymodels)
library(embed)
library(mlr3verse)
library(mlr3hyperband)
library(progressr)
library(xgboost)

options(scipen=999)


#Data prep ####

##Load and inspect data####
df <- read.csv("./OriginalData/clients.csv", header=TRUE)
summary(data.frame(unclass(df), stringsAsFactors = TRUE))
#factors: month, sex, education,  product_type, having_children_flg, region
  #, family_status, phone_operator, is_client, bad_client_target
#all columns are possible features
#no missing values


##Clean data ####


#rename columns
names(df)[8] <- "have_children"
names(df)[13] <- "prev_client"
names(df)[14] <- "bad_client"




#rename factor levels, absorb unfrequent levels
#education
df$education <- recode(df$education, "Higher education"="Higher",
                       "Incomplete higher education"="HigherDropout",
                       "Incomplete secondary education"="Secondary",
                       "PhD degree"="Higher",
                       "Secondary education"="Secondary",
                       "Secondary special education"="SecondarySpec")

#product_type
prod_types <- as.data.frame(table(as.factor(df$product_type)))
#group those with less than 80 obs:
prod_types_rare <- subset(prod_types, Freq <=80)
#master groups:
  #fashion: Cosmetics and beauty services, jewelry
  #vehicles: Auto, Boats
  #recreational: Tourism, Music, Fishing and hunting supplies, Sporting goods, Audio & Video,
    #Fitness 
  #hardware: Windows & Doors, Construction Materials, Repair Services, Garden equipment
  #necessities: Medical services, Training, Children's goods
df$product_type <- recode(df$product_type, 
                          "Cosmetics and beauty services" = "Fashion",
                          "Jewelry" = "Fashion",
                          "Auto" = "Vehicles",
                          "Boats" = "Vehicles"
                          )

lookup_recreational <- tibble(old=c("Tourism", "Music", "Fishing and hunting supplies",
                                    "Sporting goods", "Audio & Video", "Fitness"),
                              new=c(rep("Recreational", 6)))
df$product_type <- recode(df$product_type, !!!deframe(lookup_recreational))


lookup_hardware <- tibble(old=c("Windows & Doors", "Construction Materials", "Repair Services",
                                "Garden equipment"),
                              new=c(rep("Hardware", 4)))
df$product_type <- recode(df$product_type, !!!deframe(lookup_hardware))

df$product_type <- recode(df$product_type, 
                          "Medical services" = "Necessities",
                          "Training" = "Necessities",
                          "Childen's goods" = "Necessities")

df$product_type <- recode(df$product_type, "Fashion"="Beauty")

df$product_type <- recode(df$product_type, "Household appliances"="Appliances")

rm(prod_types, prod_types_rare, lookup_hardware, lookup_recreational)




#factorize factors. only ordinal: education

#month, sex, product_type, family status, region, phone operator: 
  #factorize, don't change levels
for (i in c(1, 5, 7, 9, 11, 12)){
  df[,i] <- as.factor(df[,i])
}
rm(i)


#education: factorize, order levels
summary(as.factor(df$education))
df$education <- factor(df$education, levels=c("SecondarySpec", "Secondary",
                                              "HigherDropout", "Higher"),
                       ordered=TRUE)


#have children, region, phone_operator, prev_client, bad_client
  #recode values, factorize
for (i in c(8, 13, 14)) {
  df[,i] <- recode(as.character(df[,i]),
                   "0" = "No",
                   "1" = "Yes")
  df[,i] <- as.factor(df[,i])
}
rm(i)




#summary of results
summary(df)






#Exploratory analysis ####  


##Numerics ####
ggpairs(df,
        columns=c("credit_amount", "credit_term", "age", "income", "bad_client"),
        mapping=(aes(color=bad_client, fill=bad_client, alpha=0.5)))
#credit_amount:
  #bad clients have slightly higher median credit amount, but also fewer high outliers
  #0.5 correlation with credit_term
  #very right tailed distribution
#credit_term:
  #considerably higher for bad clients
  #0.5 correlation with credit_amount
  #bit more positively correlated with income for bad clients, compared to all
  #right tailed distribution
#age:
  #considerably lower for bad clients
  #bit more positively correlated with income for bad clients
  #right tailed distribution
#income:
  #lower for bad clients
  #a little correlated with credit amount, 0.373
  #bit more positively correlated with credit_term and age for bad clients
  #very right tailed distribution




##Categoricals ####
plot_bar(df, by="bad_client")
#all categoricals have significant relationship with bad_client

plot_bar(df)
#region 1 and phone operator 4 are very rare




##Interactions ####


#possible interactions:
  #credit_amount X product type
  #age X income
  #age X family_status
  #have_children X family_status


#test interactions:

#credit amount X product_type
ggplot(df, aes(product_type, credit_amount, fill=bad_client)) + 
  geom_boxplot()
#for some product types, higher credit amount is associated much more with bad clients
  #example: high credit amount for clothing: bad
#for some, lower credit amount is associated with good clients
  #example: high credit amount for recreation: good
#add interaction


#age X income
ggplot(df, aes(age, income, color=bad_client)) + geom_point() +
  geom_smooth()
#good clients' income peaks around 30, then declines
#bad clients' income peaks around 45, then declines
#small but significant interaction. add


#age X family_status
ggplot(df, aes(family_status, age, fill=bad_client)) + 
  geom_boxplot()
#for married and another, bad clients are younger
#for unmarried, bad clients are a bit older
#add interaction


#age X have_children
ggplot(df, aes(have_children, age, fill=bad_client)) + 
  geom_boxplot()
#without kids, bad clients are much younger
#with kids, bad and good clients are almost at the same age
#add interaction



#have_children X family_status
plot(x=df$family_status, y=df$bad_client)
plot(x=df$family_status:df$have_children, y=df$bad_client)
#bad clients tend to be married > unmarried > another, regardless of children
#no significant interaction


##Summary of EDA ####

#use all original features
#log transform numeric features
#no strong multicollinearity
#interactions to add:
  #credit_amount X product_type
  #age X income
  #age X family_status
  #age X have_children
  




#Feature engineering ####


##Creating features ####

#payment: amount to be paid in 1 period
df <- df %>% mutate(payment = round(credit_amount / credit_term, 0), .after=credit_term)


#ratio_amount: ratio of amount to be paid and income
df <- df %>% mutate(ratio_amount = round(credit_amount / income, 2), .after=credit_amount)


#ratio_payment: ratio of installment to income
df <- df %>% mutate(ratio_payment = round(payment / income, 2), .after=payment)




##Evaluating features ####
ggpairs(df,
        columns=c("payment", "ratio_amount", "ratio_payment", "bad_client"),
        mapping=(aes(color=bad_client, fill=bad_client, alpha=0.5)))
#payment:
  #lower for bad clients
  #positively correlated  with ratio_amount (0.5) and ratio payment (0.732), for bad clients
  #very strong right tail distribution
#ratio_amount:
  #lower for bad clients
  #strong positive correlation with ratio_payment (0.84), but less so for bad clients (0.56)
  #very strong right tail distribution
#ratio_payment:
  #same as ratio_amount
  #lower for bad clients




##Summary of FeatureEng ####

#keep payment
#keep both ratios, or decide between one, or PCA them?






#Preprocessing ####


#ordinal encode education manually
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL)) - 1
  x
}

df$education <- encode_ordinal(df$education, order=levels(df$education))


#create preprocessing pipeline

#boxcox numerics
pipe_yeo <- po("yeojohnson", standardize=FALSE)

#woe encode nominals
encode_woe <- po("encodeimpact")

#center and scale numerics
pipe_center_scale <- po("scale")

#add interactions:
    #credit_amount X product_type
    #age X income
    #age X family_status
    #age X have_children
pipe_interact <- po("modelmatrix", formula = ~ . + credit_amount:product_type.No +
                      credit_amount:product_type.Yes +
                      age:income + age:family_status.No + age:family_status.Yes +
                      age:have_children.No + age:have_children.Yes)

#add class weights
pipe_weights <- po("classweights")
pipe_weights$param_set$values$minor_weight <- summary(df$bad_client)[1] /
  summary(df$bad_client)[2]


#create preprocessing graph
graph_preproc <- pipe_yeo %>>% encode_woe %>>% pipe_center_scale %>>%
  pipe_interact %>>% pipe_weights




#Modeling####


#task
task_credit <- as_task_classif(df, target="bad_client", positive="Yes")


#resampling: 5 folds
resample_cv <- rsmp("cv", folds=5L)


#performance measures
source("mlr3CustomMeasures.R")

#evaluation metrics
measure_rec <- msr("classif.recall") 
measure_prauc <- msr("classif.prauc") 
measure_brier <- msr("classif.bbrier") 
measure_kappa <- msr("classif.kappa") 
measure_acc <- msr("classif.acc") 
measure_time <- msr("time_both")
measures_list <- c(measure_rec, measure_prauc, measure_brier, measure_kappa, 
                   measure_acc, measure_time)

#logloss just for tuning
measure_log <- msr("classif.logloss")


#tuners
tune_grid <- mlr3tuning::tnr("grid_search")
tune_random <- mlr3tuning::tnr("random_search") 
tune_hyper2 <- mlr3tuning::tnr("hyperband", eta=2) 
tune_hyper3 <- mlr3tuning::tnr("hyperband", eta=3) 
tune_anneal <- mlr3tuning::tnr("gensa") 


#terminators
term10 <- trm("evals", n_evals=10)
term25 <- trm("evals", n_evals=25)
term50 <- trm("evals", n_evals=50)
term100 <- trm("evals", n_evals=100)
term250 <- trm("evals", n_evals=250)
term500 <- trm("evals", n_evals=500)
term_none <- trm("none")


#Bayes graph learner
learner_bayes <- as_learner(graph_preproc %>>% po("learner", 
                                                  learner=lrn("classif.naive_bayes",
                                                  predict_type="prob")))
#note: the nominal encoding may impact the performance



##Tune models ####


###glmnet ####


#graph learner
learner_glmnet <- as_learner(graph_preproc %>>% po("learner", 
                                                   learner=lrn("classif.glmnet",
                                                   predict_type="prob")))



#tuning space
space_glmnet <- ps(
  classif.glmnet.alpha=p_dbl(lower=0, upper=1),
  classif.glmnet.lambda=p_dbl(lower=0, upper=1)
)



#auto tuning
autotune_glmnet = AutoTuner$new(
  learner=learner_glmnet,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term100,
  tuner=tune_anneal,
  search_space=space_glmnet
)


#tune glmnet
set.seed(1922)
with_progress(autotune_glmnet$train(task_credit))
#alpha 0.1231044 lambda 0.04147188 logloss 0.5871771

#extract archive
archive_glmnet <- as.data.table(autotune_glmnet$archive)

#set best params to glmnet graph learner
learner_glmnet$param_set$values <- autotune_glmnet$learner$param_set$values




###kNN ####


#graph learner
learner_knn <- as_learner(graph_preproc %>>% po("learner", 
                                                   learner=lrn("classif.kknn",
                                                               predict_type="prob",
                                                               kernel="optimal",
                                                               scale=FALSE)))



#search space

#trafo function
k_square_odd <- function(k){
  k <- 2^k + 1
  k
}

#tuning space
space_knn = ps(
  classif.kknn.k=p_int(1, 10, trafo=k_square_odd, tags="budget"),
  classif.kknn.distance=p_dbl(1, 2)
)



#auto tuning
autotune_knn = AutoTuner$new(
  learner=learner_knn,
  resampling=resample_cv,
  measure=measure_prauc,
  terminator=term_none,
  tuner=tune_hyper2,
  search_space=space_knn
)


#tune knn
set.seed(1922)
with_progress(autotune_knn$train(task_credit))

#extract archive
archive_knn <- as.data.table(autotune_knn$archive)
#k 1025, dist 1.843605, prauc 0.2431451





#tuning space 2, 34-1025
k_times33 <- function(k) {
  k <- k*32 + 1
  k
}


#tuning space
space_knn2 = ps(
  classif.kknn.k=p_int(1, 30, trafo=k_times33, tags="budget"),
  classif.kknn.distance=p_dbl(1, 2)
)



#auto tuning
autotune_knn2 = AutoTuner$new(
  learner=learner_knn,
  resampling=resample_cv,
  measure=measure_prauc,
  terminator=term_none,
  tuner=tune_hyper2,
  search_space=space_knn2
)



#tune knn2
set.seed(1922)
with_progress(autotune_knn2$train(task_credit))
#


#extract archive
archive_knn2 <- as.data.table(autotune_knn2$archive)
#k 129, dist 1.843605, prauc 0.2504462


#set best params to knn graph learner
learner_knn$param_set$values <- autotune_knn2$learner$param_set$values





###SVM ####

#named class weights vector
svm_weight <- c(1, learner_knn$param_set$values$classweights.minor_weight)
names(svm_weight) <- c("No", "Yes")


#graph learner
learner_svm <- as_learner(graph_preproc %>>% po("learner", 
                                                learner=lrn("classif.svm",
                                                            type="C-classification",
                                                            class.weights=svm_weight,
                                                            predict_type="prob",
                                                            kernel="radial",
                                                            scale=FALSE)))


#trafo functions
k_square <- function(k) {
  k <- 2^k
  k
}

k_power10 <- function(k) {
  k <- 10^k
  k
}


#tuning space
space_svm = ps(
  classif.svm.cost=p_int(1, 6, trafo=k_power10, tags="budget"),
  classif.svm.gamma=p_int(-8, 2, trafo=k_square)
)



#auto tuning
autotune_svm = AutoTuner$new(
  learner=learner_svm,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term_none,
  tuner=tune_hyper2,
  search_space=space_svm
)


#tune svm
set.seed(1922)
with_progress(autotune_svm$train(task_credit))

#extract archive
archive_svm <- as.data.table(autotune_svm$archive)
#cost 100, gamma 0.0156250, logloss 0.3411680




#tuning space 2, C from 1-100, gamma 0-0.1
k_10fold <- function(k) {
  k <- k*10
  k
}


#tuning space
space_svm2 = ps(
  classif.svm.cost=p_int(1, 10, trafo=k_10fold, tags="budget"),
  classif.svm.gamma=p_dbl(0, 0.1)
)



#auto tuning
autotune_svm2 = AutoTuner$new(
  learner=learner_svm,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term_none,
  tuner=tune_hyper2,
  search_space=space_svm2
)


#tune svm2
set.seed(1922)
with_progress(autotune_svm2$train(task_credit))

#extract archive
archive_svm2 <- as.data.table(autotune_svm2$archive)
#cost 10, gamma 0.009167361, logloss 0.3302667

#set best params to svm graph learner
learner_svm$param_set$values <- autotune_svm2$learner$param_set$values






###XGBoost ####


#graph learner
learner_xgb <- as_learner(graph_preproc %>>% po("learner", 
                                                learner=lrn("classif.xgboost",
                                                            booster="gbtree",
                                                            predict_type="prob",
                                                            nthread=3,
                                                            nrounds=5000,
                                                            early_stopping_rounds=50)))



####tune eta ####

#search space
space_xgb1 = ps(
  classif.xgboost.eta=p_dbl(0.01, 0.05)
)


#auto tuning
autotune_xgb1 = AutoTuner$new(
  learner=learner_xgb,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term25,
  tuner=tune_anneal,
  search_space=space_xgb1
)


#tune xgb1
set.seed(1922)
with_progress(autotune_xgb1$train(task_credit))

#extract archive
archive_xgb1 <- as.data.table(autotune_xgb1$archive)
#eta 0.01013721, logloss 0.553274

#retrieve eta
learner_xgb$param_set$values <- autotune_xgb1$learner$param_set$values




####tune tree complexity ####

#trafo
k_twofold <- function(k) {
  k <- k*2
  k
}

#search space
space_xgb2 = ps(
  classif.xgboost.max_depth=p_int(1, 10, trafo=k_twofold),
  classif.xgboost.min_child_weight=p_int(1, 20, trafo=k_twofold, tags="budget"),
  classif.xgboost.max_delta_step=p_int(0, 5, trafo=k_twofold)
)


#auto tuning
autotune_xgb2 = AutoTuner$new(
  learner=learner_xgb,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term_none,
  tuner=tune_hyper2,
  search_space=space_xgb2
)


#tune xgb2
set.seed(1922)
with_progress(autotune_xgb2$train(task_credit))

#extract archive
archive_xgb2 <- as.data.table(autotune_xgb2$archive)
#max depth=12  min_child_weight=20  max_delta_step=8 logloss 0.4427219

#retrieve values
learner_xgb$param_set$values <- autotune_xgb2$learner$param_set$values




####tune regularization ####


#search space
space_xgb3 = ps(
  classif.xgboost.gamma=p_dbl(0, 1),
  classif.xgboost.lambda=p_dbl(0, 2),
  classif.xgboost.alpha=p_dbl(0, 1)
)


#auto tuning
autotune_xgb3 = AutoTuner$new(
  learner=learner_xgb,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term25,
  tuner=tune_anneal,
  search_space=space_xgb3
)


#tune xgb3
set.seed(1922)
with_progress(autotune_xgb3$train(task_credit))

#extract archive
archive_xgb3 <- as.data.table(autotune_xgb3$archive)
#gamma=0.73399971  lambda=0.9294227 alpha=0.14594723  logloss=0.4264966

#retrieve regularization
learner_xgb$param_set$values <- autotune_xgb3$learner$param_set$values





####tune randomization ####


#search space
space_xgb4 = ps(
  classif.xgboost.subsample=p_dbl(0.5, 1),
  classif.xgboost.colsample_bytree=p_dbl(0.5, 1)
)


#auto tuning
autotune_xgb4 = AutoTuner$new(
  learner=learner_xgb,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term10,
  tuner=tune_anneal,
  search_space=space_xgb4
)


#tune xgb4
set.seed(1922)
with_progress(autotune_xgb4$train(task_credit))

#extract archive
archive_xgb4 <- as.data.table(autotune_xgb4$archive)
#interrupted: not helping






#### retune eta ####


#search space
space_xgb5 = ps(
  classif.xgboost.eta=p_dbl(0.01, 0.2)
)


#auto tuning
autotune_xgb5 = AutoTuner$new(
  learner=learner_xgb,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term25,
  tuner=tune_anneal,
  search_space=space_xgb5
)


#tune xgb5
set.seed(1922)
with_progress(autotune_xgb5$train(task_credit))

#extract archive
archive_xgb5 <- as.data.table(autotune_xgb5$archive)
#eta=0.06444704  logloss=0.4259129  

#retrieve eta
learner_xgb$param_set$values <- autotune_xgb5$learner$param_set$values





#### tune nrounds ####


#retrieve preprocessed data
task_trained <- graph_preproc$train(task_credit)
df_train <- task_trained$classweights.output$data()
x_train <- df_train[,-c(1)]
y_train <- df_train[,1]
y_train <- c(as.numeric(y_train$bad_client)-1)
y_train <- abs(y_train-1)
weight_xgb <- task_trained$classweights.output$weights$weight


#create XGB data matrix
xgb_train <- xgb.DMatrix(data=as.matrix(x_train), 
                         label=y_train,
                         weight=weight_xgb)


#create xgb params list
xgb_params <- list (
  booster="gbtree",
  objective="binary:logistic",
  nthread=3,
  max_delta_step=8,
  eta=0.06444704,
  max_depth=12,
  min_child_weight=20,
  gamma=0.73399971,
  lambda=0.9294227,
  alpha=0.14594723
)


#run xgb.cv
set.seed(1922)
xgb_cv <- xgb.cv(xgb_params, xgb_train, nfold=5, verbose=TRUE, nrounds=5000,
                   early_stopping_rounds = 50)
#27 rounds
#train-logloss:0.430903+0.003545
#test-logloss:0.625221+0.033639
#U-shaped crossvalidation



#final learner with best tune
learner_xgb$param_set$values$classif.xgboost.nrounds <- 27


  
  
  
  
## Benchmarking ####


### Perform benchmarking ####

#create baseline learner that predicts the class prob distributions
learner_baseline <- as_learner(graph_preproc %>>% 
                                 po("learner", 
                                    learner=lrn("classif.featureless",
                                    predict_type="prob",
                                    method="weighted.sample")))
  
  
  



#create benchmark grid
benchmark_test = benchmark_grid(tasks=task_credit,
                                   learn=list(learner_baseline, learner_bayes, 
                                              learner_glmnet,
                                              learner_knn, learner_svm,
                                              learner_xgb),
                                   resamplings=resample_cv)



#perform benchmarking
set.seed(1923)
benchmarkres = benchmark(benchmark_test, store_models=TRUE)


#save average benchmarking results
benchmarkres_table <- benchmarkres$aggregate(measures_list)

#save all benchmarking round results
benchmarkres_all <- benchmarkres$score(measures_list)



### Plots and tables ####


#prc curves
prc_plot <- mlr3viz::autoplot(benchmarkres, type="prc")
prc_plot <- prc_plot + 
  geom_line(aes(x=x, y=y, color=modname)) +
  geom_ribbon(aes(x=x, y=y, ymin=0, ymax=y, fill=modname), alpha=0.075) +
  labs(x="Recall", y="Precision",
       title="Precision-recall curves of loan client classifiers",
       subtitle="Baseline classifier predicts the class proportions as the probability of each class",
       color="Classifiers",
       fill="Classifiers") + 
  theme(legend.position = "right") + 
  scale_color_brewer(palette="Dark2", labels=c("Baseline", "NaiveBayes", "glmnet", "kNN", "SVM", "XGBoost")) + 
  scale_fill_brewer(palette="Dark2", labels=c("Baseline", "NaiveBayes", "glmnet", "kNN", "SVM", "XGBoost")) +
  theme_bw()


#line size
prc_plot$layers[[1]]$aes_params$size=0.75

#supress confidence bands
prc_plot <- remove_geom(p=prc_plot, geom="smooth")

prc_plot




#table of metrics
df_metrics <- benchmarkres_table[,c(4, 7:12)]
colnames(df_metrics) <- c("Classifier", "Recall", "PRAUC",
                              "Brier score", "Cohen's kappa", "Accuracy", "Time")
df_metrics[1:6, 1] <- c("Baseline", "NaiveBayes", "glmnet", "kNN", "SVM", "XGBoost")
df_metrics[,2:6] <- round(df_metrics[,2:6], 4)


tb_perf <- gt(data=df_metrics, rowname_col = 1) %>% 
  tab_header(title="Performance metrics of loan client classifiers",
             subtitle="Tested on previously unseen data. Threshold probability=0.5") %>% 
  opt_table_font(font=list(google_font("Calibri"), default_fonts())) %>% 
  cols_align(align = "center", columns = everything()) %>% 
  tab_style(locations=cells_column_labels(columns=everything()), 
            style=list(
              cell_borders(sides="bottom", weight=px(3)), 
              cell_text(weight="bold"))) %>% 
  data_color(columns=c(2,3,5,6), colors=scales::col_numeric(
    palette=c("red", "green"),
    domain=c(-0.0023,1)
  )) %>%
  data_color(columns=c(4), colors=scales::col_numeric(
    palette=c("green", "red"),
    domain=c(0,2)
    )) %>%
      data_color(columns=c(7), colors=scales::col_numeric(
        palette=c("green", "red"),
        domain=c(0,max(df_metrics[,7]))
        ))

tb_perf
