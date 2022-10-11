#LOAN REQUEST CLASSIFICATION

#Libraries and options ####
library(tidyverse)
library(GGally)
library(DataExplorer)
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


#split train-test data
set.seed(1923)
data_split <- initial_split(df, prop=4/5, strata=bad_client)
df_train <- training(data_split)
df_test <- testing(data_split)
rm(data_split)


#create preprocessing recipe
  #log transform 
  #ordinal encode education
  #nominal encode all other categoricals
  #center and scale
  #PCA ratio_amount and ratio_payment
  #add interactions:
    #credit_amount X product_type
    #age X income
    #age X family_status
    #age X have_children
recipe0 <- recipe(bad_client ~ ., data=df_train) %>%
  step_log(all_numeric_predictors(), signed=TRUE) %>%
  step_ordinalscore(education) %>%
  step_woe(all_nominal_predictors(), outcome="bad_client") %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_pca(ratio_amount, ratio_payment, num_comp=2) %>%
  step_interact(terms = ~ credit_amount:woe_product_type + age:income + 
age:woe_family_status + age:woe_have_children)


#prep recipe on training data
recipe0_prep <- prep(recipe0, training=df_train)


#bake recipe on training and testing data
df_train <- bake(recipe0_prep, new_data=df_train)
df_test <- bake(recipe0_prep, new_data=df_test)


#move outcome columns to start
df_train <- df_train %>% relocate(bad_client, .before=credit_amount)
df_test <- df_test %>% relocate(bad_client, .before=credit_amount)


#create train_x and test_x
train_x <- df_train[,-c(1)]
test_x <- df_test[,-c(1)]


#create numeric outcome vectors
train_y <- as.numeric(df_train$bad_client) - 1 
test_y <- as.numeric(df_test$bad_client) - 1


#retrieve minority weight from training data
minority_weight <- length(df_train$bad_client[df_train$bad_client=="No"]) /
  length(df_train$bad_client[df_train$bad_client=="Yes"])


#bind back training and testing data for mlr3 resampling
df1 <- rbind(df_train, df_test)
rm(recipe0, recipe0_prep)


#test if the PCs of ratios are predictive
ggplot(data=df_train, aes(x=bad_client, y=PC1)) + geom_boxplot()
#PC1 similar to original ratio features
ggplot(data=df_train, aes(x=bad_client, y=PC2)) + geom_boxplot()
#PC2 similar to original ratio features


#test distributions of transformed features
plot_histogram(df_train, binary_as_factor = FALSE)
#original numerics normalish
#PCs not normal
#education (ordinal) not normal
#WOEs and their interactions not normal
#age X income not normal, age X woe family and children normalish
#credit X woe product not normal




#Modeling####


##Create common objects####


#tasks
task_train <- as_task_classif(df_train, target="bad_client", positive="Yes")
task_test <- as_task_classif(df1, target="bad_client", positive="Yes")


#assign class weights
pipe_weights <- po("classweights")
pipe_weights$param_set$values$minor_weight <- minority_weight
task_train <- pipe_weights$train(list(task_train))[[1L]]
task_test <- pipe_weights$train(list(task_test))[[1L]]


#resamplings
#5 fold, 5 repeats for tuning
resample_cv <- rsmp("repeated_cv", folds=5L, repeats=5L)

#custom resampling for testing
id_train <- list(task_train$row_ids)
id_test <- list(setdiff(task_test$row_ids, task_train$row_ids))
resample_test <- rsmp("custom")
resample_test$instantiate(task_test, train_sets=id_train, test_sets=id_test)


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


#Bayes learner
learner_bayes <- lrn("classif.naive_bayes",
                     predict_type="prob")
#note: the nominal encoding may impact the performance



##Tune models ####


###glmnet ####

#tuning space
learner_glmnet1 <- lrn("classif.glmnet",
                       predict_type="prob",
                       alpha=to_tune(p_dbl(0,1)),
                       lambda=to_tune(p_dbl(0,1)))


#tuning instance
tune_glmnet1 = TuningInstanceSingleCrit$new(task_train, learner_glmnet1, resample_cv, 
                                           measure_log, term100,
                                           store_benchmark_result = TRUE)

#tune glmnet
tune_anneal$initialize()
set.seed(1923)
with_progress(tune_anneal$optimize(tune_glmnet1))


tuneres_glmnet1 <- as.data.table(tune_glmnet1$archive)
#tried any alpha from 0-0.6
#tried any lambda from 0.0017 to 0.9
#best: alpha 0.55389688 lambda 0.0047062116 logloss 0.5811211


#set learner parameters to best tune
learner_glmnet <- lrn("classif.glmnet",
                      predict_type="prob",
                      alpha=0.55389688,
                      lambda=0.0047062116)




###kNN ####

#trafo function
k_square_odd <- function(k){
  k <- 2^k + 1
  k
}

#tuning space
learner_knn1 <- lrn("classif.kknn",
                    predict_type="prob",
                    kernel="optimal",
                    scale=FALSE,
                    k=to_tune(p_int(1, 10, trafo=k_square_odd, tags="budget")),
                    distance=to_tune(p_dbl(1, 2))
                    )


#tuning instance
tune_knn1 = TuningInstanceSingleCrit$new(task_train, learner_knn1, resample_cv, 
                                            measure_prauc, term_none,
                                            store_benchmark_result = TRUE)

#tune glmnet
tune_hyper2$initialize()
set.seed(1923)
with_progress(tune_hyper2$optimize(tune_knn1))


tuneres_knn1 <- as.data.table(tune_knn1$archive)
#k 1025, distance 1.003422, prauc 0.2471221
#try 34-1025




#tuning space 2, 34-1025
k_times33 <- function(k) {
  k <- k*32 + 1
  k
}

learner_knn2 <- lrn("classif.kknn",
                    predict_type="prob",
                    kernel="optimal",
                    scale=FALSE,
                    k=to_tune(p_int(1, 30, trafo=k_times33, tags="budget")),
                    distance=to_tune(p_dbl(1, 2))
)


#tuning instance
tune_knn2 = TuningInstanceSingleCrit$new(task_train, learner_knn2, resample_cv, 
                                         measure_prauc, term_none,
                                         store_benchmark_result = TRUE)

#tune glmnet
tune_hyper2$initialize()
set.seed(1923)
with_progress(tune_hyper2$optimize(tune_knn2))

tuneres_knn2 <- as.data.table(tune_knn2$archive)
#k 257, distance 1.003422, prauc 0.2529354


#set learner parameters to best tune
learner_knn <- lrn("classif.kknn",
                   predict_type="prob",
                   kernel="optimal",
                   k=257,
                   distance=1.003422)





###SVM ####

#trafo functions
k_square <- function(k) {
  k <- 2^k
  k
}

k_power10 <- function(k) {
  k <- 10^k
  k
}

#named class weights vector
svm_weight <- c(1, minority_weight)
names(svm_weight) <- c("No", "Yes")


#tuning space
learner_svm1 <- lrn("classif.svm",
                    type="C-classification",
                    class.weights=svm_weight,
                    predict_type="prob",
                    kernel="radial",
                    scale=FALSE,
                    cost=to_tune(p_int(1, 6, trafo=k_power10, tags="budget")),
                    gamma=to_tune(p_int(-8, 2, trafo=k_square)))



#tuning instance
tune_svm1 = TuningInstanceSingleCrit$new(task_train, learner_svm1, resample_cv, 
                                         measure_log, term_none,
                                         store_benchmark_result = TRUE)

#tune glmnet
tune_hyper2$initialize()
set.seed(1923)
with_progress(tune_hyper2$optimize(tune_svm1))

#results
tuneres_svm1 <- as.data.table(tune_svm1$archive)
#cost 1000, gamma 4, logloss 0.3713911
#try costs around 1000, gamma>4




#tuning space 2, C around 1000, gamma>4
k_150fold <- function(k) {
  k <- k*150
  k
}


learner_svm2 <- lrn("classif.svm",
                    type="C-classification",
                    class.weights=svm_weight,
                    predict_type="prob",
                    kernel="radial",
                    scale=FALSE,
                    cost=to_tune(p_int(1, 10, trafo=k_150fold, tags="budget")),
                    gamma=to_tune(p_dbl(4, 8)))



#tuning instance
tune_svm2 = TuningInstanceSingleCrit$new(task_train, learner_svm2, resample_cv, 
                                         measure_log, term_none,
                                         store_benchmark_result = TRUE)

#tune glmnet
tune_hyper2$initialize()
set.seed(1923)
with_progress(tune_hyper2$optimize(tune_svm2))

#results
tuneres_svm2 <- as.data.table(tune_svm2$archive)
#cost 750 gamma 4.916431 logloss 0.3718737




#learner with best parameters
learner_svm <- lrn("classif.svm",
                    type="C-classification",
                    class.weights=svm_weight,
                    predict_type="prob",
                    kernel="radial",
                    scale=FALSE,
                    cost=750,
                    gamma=4.916431)







###XGBoost ####


####tune eta, max_delta_step ####

#tuning space
learner_xgb1 <- lrn("classif.xgboost",
                   booster="gbtree",
                   predict_type="prob",
                   nthread=3,
                   nrounds=5000,
                   early_stopping_rounds=50,
                   max_delta_step=to_tune(p_int(1,10, tags="budget")),
                   eta=to_tune(p_dbl(0.1, 0.3)))



#tuning instance
tune_xgb1 = TuningInstanceSingleCrit$new(task_train, learner_xgb1, resample_cv, 
                                         measure_log, term_none,
                                         store_benchmark_result = TRUE)

#tune xgboost1
tune_hyper2$initialize()
set.seed(1923)
with_progress(tune_hyper2$optimize(tune_xgb1))

#results
tuneres_xgb1 <- as.data.table(tune_xgb1$archive)
#eta 0.1081076, max_delta_step 1, logloss 0.6996111
#bigger max delta steps are worse




#check against max_delta_step = 0

#tuning space
learner_xgb1.1 <- learner_xgb1
learner_xgb1.1$param_set$values$max_delta_step <- 0



#tuning instance
tune_xgb1.1 = TuningInstanceSingleCrit$new(task_train, learner_xgb1.1, resample_cv, 
                                         measure_log, term10,
                                         store_benchmark_result = TRUE)

#tune xgboost1.1
tune_anneal$initialize()
set.seed(1923)
with_progress(tune_anneal$optimize(tune_xgb1.1))

#results
tuneres_xgb1.1 <- as.data.table(tune_xgb1.1$archive)
#eta 0.1, max_delta_step 0, logloss 0.6789039





####tune tree complexity ####


#tuning space
k_twofold <- function(k){
  k <- k*2
  k
}

learner_xgb2 <- learner_xgb1
learner_xgb2$param_set$values$max_delta_step <- 0
learner_xgb2$param_set$values$eta <- 0.1
learner_xgb2$param_set$values$max_depth <- to_tune(p_int(1, 6, 
                                                         trafo=k_twofold))
learner_xgb2$param_set$values$min_child_weight <- to_tune(p_int(1, 10,
                                                                tags="budget"))



#tuning instance
tune_xgb2 = TuningInstanceSingleCrit$new(task_train, learner_xgb2, resample_cv, 
                                         measure_log, term_none,
                                         store_benchmark_result = TRUE)

#tune xgboost2
tune_hyper2$initialize()
set.seed(1923)
with_progress(tune_hyper2$optimize(tune_xgb2))

#results
tuneres_xgb2<- as.data.table(tune_xgb2$archive)
#12 max_depth, 1 min_child_weight, logloss 0.6404987
#12 max_depth, 2 min_child_weight, logloss 0.6087693
#12 max depth, 5 min_child_weight, logloss 0.5768753
  #very close to max_depth 10's performance. don't try higher max_depth?
#best tune: max depth 12, min child weight 10, logloss 0.5579934
  #try higher



####tune regularization ####


#tuning space
learner_xgb3 <- learner_xgb2
learner_xgb3$param_set$values$max_depth <- #???
learner_xgb3$param_set$values$min_child_weight <- #???
learner_xgb3$param_set$values$gamma <- to_tune(p_int(0, 1))
learner_xgb3$param_set$values$lambda <- to_tune(p_dbl(0, 2))
learner_xgb3$param_set$values$alpha <- to_tune(p_dbl(0, 1))



#tuning instance
tune_xgb3 = TuningInstanceSingleCrit$new(task_train, learner_xgb3, resample_cv, 
                                         measure_log, term100,
                                         store_benchmark_result = TRUE)

#tune xgboost3
tune_anneal$initialize()
set.seed(1923)
with_progress(tune_anneal$optimize(tune_xgb3))

#results
tuneres_xgb3 <- as.data.table(tune_xgb3$archive)





####tune randomization ####


#tuning space
learner_xgb4 <- learner_xgb3
learner_xgb4$param_set$values$gamma <- #???
learner_xgb4$param_set$values$lambda <- #???
learner_xgb4$param_set$values$alpha <- #???
learner_xgb4$param_set$values$subsample <- to_tune(p_dbl(0.5, 1))
learner_xgb4$param_set$values$colsample_bytree <- to_tune(p_dbl(0.5, 1))



#tuning instance
tune_xgb4 = TuningInstanceSingleCrit$new(task_train, learner_xgb4, resample_cv, 
                                         measure_log, term100,
                                         store_benchmark_result = TRUE)

#tune xgboost4
tune_anneal$initialize()
set.seed(1923)
with_progress(tune_anneal$optimize(tune_xgb4))

#results
tuneres_xgb4 <- as.data.table(tune_xgb4$archive)




#### retune eta ####

#if necessary





#### tune nrounds ####


#create weights vector
xgb_weights <- c()

for (i in 1:nrow(df_train)) {
  if (df_train[i,1] == "Yes") {
    xgb_weights <- append(xgb_weights, minority_weight)
  } else {
    xgb_weights <- append(xgb_weights, 1)
  }
}


#create XGB data matrix
xgb_train <- xgb.DMatrix(data=as.matrix(train_x), label=train_y,
                         weight=xgb_weights)


#create xgb params list
xgb_params <- list (
  booster="gbtree",
  objective="binary:logistic",
  nthread=3,
  max_delta_step=,
  eta=,
  max_depth=,
  min_child_weight=,
  subsample=,
  colsample_bytree=,
  gamma=,
  lambda=,
  alpha=,
)


#run xgb.cv
set.seed(1923)
xgb_cv <- xgb.cv(xgb_params, xgb_data, nfold=5, verbose=TRUE, nrounds=?###,
                   early_stopping_rounds = 50)




#final learner with best tune
learner_xgb <- learner_xgb4
learner_xgb$param_set$values$subsample <- #???
learner_xgb$param_set$values$colsample_bytree <- #???
learner_xgb$param_set$values$nrounds <- #???


  
  
  
  
## Benchmarking ####


### Perform benchmarking ####

#create baseline learner that predicts the class prob distributions
learner_baseline <- lrn("classif.featureless",
                        predict_type="prob",
                        method="weighted.sample")


#create benchmark grid
benchmark_test = benchmark_grid(tasks=task_test,
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




### Plots and tables ####


#prc curves
prc_plot <- mlr3viz::autoplot(benchmarkres, type="prc")
prc_plot <- prc_plot + 
  geom_ribbon(aes(x=x, y=y, ymin=0, ymax=y, fill=modname), alpha=0.075) +
  labs(x="Recall", y="Precision",
       title="Precision-recall curves of loan client classifiers",
       subtitle="Featureless classifier predicts the class proportions as the probability of each class",
       color="Classifiers",
       fill="Classifiers") + 
  theme(legend.position = "right") + 
  scale_color_brewer(palette="Dark2", labels=c("Featureless", "NaiveBayes", "glmnet", "kNN", "SVM", "XGBoost")) + 
  scale_fill_brewer(palette="Dark2", labels=c("Featureless", "NaiveBayes", "glmnet", "kNN", "SVM", "XGBoost")) +
  theme_bw()


#line size
prc_plot$layers[[1]]$aes_params$size=0.75

prc_plot




#table of metrics
df_metrics <- benchmarkres_table[,c(4, 7:12)]
colnames(df_metrics) <- c("Classifier", "Recall", "PRAUC",
                              "Brier score", "Cohen's kappa", "Accuracy", "Time")
df_metrics[1:6, 1] <- c("Featureless", "NaiveBayes", "glmnet", "kNN", "SVM", "XGBoost")
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
    domain=c(0,1)
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