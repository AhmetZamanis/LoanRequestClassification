Imbalanced Classification - Loan request dataset
================
Ahmet Zamanis
2022-10-12

-   <a href="#introduction" id="toc-introduction">Introduction</a>
-   <a href="#data-preparation" id="toc-data-preparation">Data
    preparation</a>
-   <a href="#exploratory-analysis"
    id="toc-exploratory-analysis">Exploratory analysis</a>
    -   <a href="#numeric-features" id="toc-numeric-features">Numeric
        features</a>
    -   <a href="#categorical-features"
        id="toc-categorical-features">Categorical features</a>
-   <a href="#feature-engineering" id="toc-feature-engineering">Feature
    engineering</a>
    -   <a href="#creating-new-features" id="toc-creating-new-features">Creating
        new features</a>
    -   <a href="#evaluating-new-features"
        id="toc-evaluating-new-features">Evaluating new features</a>
-   <a href="#modeling" id="toc-modeling">Modeling</a>
    -   <a href="#preprocessing" id="toc-preprocessing">Preprocessing</a>
    -   <a href="#approach-to-hyperparameter-tuning"
        id="toc-approach-to-hyperparameter-tuning">Approach to hyperparameter
        tuning</a>
    -   <a href="#classifiers-with-the-best-tunes-found"
        id="toc-classifiers-with-the-best-tunes-found">Classifiers, with the
        best tunes found</a>
    -   <a href="#benchmarking" id="toc-benchmarking">Benchmarking</a>
-   <a href="#conclusions" id="toc-conclusions">Conclusions</a>

## Introduction

## Data preparation

``` r
library(tidyverse) #data handling & ggplot2
library(GGally) #exploratory analysis
library(DataExplorer) #exploratory analysis
library(patchwork) #combining ggplot2 objects
library(gt) #data tables
#library(ggedit) #editing ggplot2 objects
library(mlr3verse) #loads various mlr3 machine learning packages
library(mlr3hyperband) #hyperband tuning algorithm with mlr3
library(xgboost) #regularized gradient boosting
```

``` r
#load and summarize original data
df <- read.csv("./OriginalData/clients.csv", header=TRUE)
summary(data.frame(unclass(df), stringsAsFactors = TRUE))
```

    ##      month        credit_amount     credit_term         age            sex     
    ##  Min.   : 1.000   Min.   :  5000   Min.   : 3.00   Min.   :18.00   female:792  
    ##  1st Qu.: 3.000   1st Qu.: 13000   1st Qu.: 6.00   1st Qu.:26.00   male  :931  
    ##  Median : 7.000   Median : 21500   Median :12.00   Median :32.00               
    ##  Mean   : 6.708   Mean   : 29265   Mean   :11.55   Mean   :35.91               
    ##  3rd Qu.:10.000   3rd Qu.: 34000   3rd Qu.:12.00   3rd Qu.:44.00               
    ##  Max.   :12.000   Max.   :301000   Max.   :36.00   Max.   :90.00               
    ##                                                                                
    ##                           education                          product_type
    ##  Higher education              :585   Cell phones                  :498  
    ##  Incomplete higher education   : 86   Household appliances         :471  
    ##  Incomplete secondary education:  5   Computers                    :178  
    ##  PhD degree                    :  3   Furniture                    :164  
    ##  Secondary education           :208   Clothing                     : 88  
    ##  Secondary special education   :836   Cosmetics and beauty services: 55  
    ##                                       (Other)                      :269  
    ##  having_children_flg     region          income         family_status 
    ##  Min.   :0.0000      Min.   :0.000   Min.   :  1000   Another  :1201  
    ##  1st Qu.:0.0000      1st Qu.:2.000   1st Qu.: 21000   Married  : 444  
    ##  Median :0.0000      Median :2.000   Median : 27000   Unmarried:  78  
    ##  Mean   :0.4283      Mean   :1.681   Mean   : 32652                   
    ##  3rd Qu.:1.0000      3rd Qu.:2.000   3rd Qu.: 38000                   
    ##  Max.   :1.0000      Max.   :2.000   Max.   :401000                   
    ##                                                                       
    ##  phone_operator    is_client      bad_client_target
    ##  Min.   :0.000   Min.   :0.0000   Min.   :0.0000   
    ##  1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:0.0000   
    ##  Median :1.000   Median :1.0000   Median :0.0000   
    ##  Mean   :1.125   Mean   :0.6048   Mean   :0.1138   
    ##  3rd Qu.:2.000   3rd Qu.:1.0000   3rd Qu.:0.0000   
    ##  Max.   :4.000   Max.   :1.0000   Max.   :1.0000   
    ## 

``` r
#rename columns
names(df)[8] <- "have_children"
names(df)[13] <- "prev_client"
names(df)[14] <- "bad_client"
```

``` r
#recode levels of education column
df$education <- recode(df$education, "Higher education"="Higher",
                       "Incomplete higher education"="HigherDropout",
                       "Incomplete secondary education"="Secondary",
                       "PhD degree"="Higher",
                       "Secondary education"="Secondary",
                       "Secondary special education"="SecondarySpec")
```

``` r
#recode levels of product_type column, combining unfrequent levels together
#new levels: Beauty, Vehicles, Necessities
df$product_type <- recode(df$product_type, 
                          "Cosmetics and beauty services" = "Beauty",
                          "Jewelry" = "Beauty",
                          "Auto" = "Vehicles",
                          "Boats" = "Vehicles",
                          "Medical services" = "Necessities",
                          "Training" = "Necessities",
                          "Childen's goods" = "Necessities"
                          )


#new level: Recreational
lookup_recreational <- tibble(old=c("Tourism", "Music", "Fishing and hunting supplies",
                                    "Sporting goods", "Audio & Video", "Fitness"),
                              new=c(rep("Recreational", 6)))
df$product_type <- recode(df$product_type, !!!deframe(lookup_recreational))


#new level: Hardware
lookup_hardware <- tibble(old=c("Windows & Doors", "Construction Materials", "Repair Services",
                                "Garden equipment"),
                              new=c(rep("Hardware", 4)))
df$product_type <- recode(df$product_type, !!!deframe(lookup_hardware))


#shorten the level Household appliances
df$product_type <- recode(df$product_type, "Household appliances"="Appliances")

#remove intermediary objects
rm(prod_types, prod_types_rare, lookup_hardware, lookup_recreational)
```

``` r
#factor convert categorical columns
#month, sex, product_type, family status, region, phone operator: 
  #factorize, don't change levels
for (i in c(1, 5, 7, 9, 11, 12)){
  df[,i] <- as.factor(df[,i])
}
rm(i)


#have children, region, phone_operator, prev_client, bad_client
  #factorize, recode levels
for (i in c(8, 13, 14)) {
  df[,i] <- recode(as.character(df[,i]),
                   "0" = "No",
                   "1" = "Yes")
  df[,i] <- as.factor(df[,i])
}
rm(i)


#education: factorize and order levels
df$education <- factor(df$education, levels=c("SecondarySpec", "Secondary",
                                              "HigherDropout", "Higher"),
                       ordered=TRUE)
```

``` r
summary(df)
```

    ##      month     credit_amount     credit_term         age            sex     
    ##  11     :174   Min.   :  5000   Min.   : 3.00   Min.   :18.00   female:792  
    ##  12     :162   1st Qu.: 13000   1st Qu.: 6.00   1st Qu.:26.00   male  :931  
    ##  10     :160   Median : 21500   Median :12.00   Median :32.00               
    ##  3      :158   Mean   : 29265   Mean   :11.55   Mean   :35.91               
    ##  7      :145   3rd Qu.: 34000   3rd Qu.:12.00   3rd Qu.:44.00               
    ##  8      :142   Max.   :301000   Max.   :36.00   Max.   :90.00               
    ##  (Other):782                                                                
    ##          education        product_type have_children region       income      
    ##  SecondarySpec:836   Cell phones:498   No :985       0: 240   Min.   :  1000  
    ##  Secondary    :213   Appliances :471   Yes:738       1:  69   1st Qu.: 21000  
    ##  HigherDropout: 86   Computers  :178                 2:1414   Median : 27000  
    ##  Higher       :588   Furniture  :164                          Mean   : 32652  
    ##                      Beauty     : 91                          3rd Qu.: 38000  
    ##                      Clothing   : 88                          Max.   :401000  
    ##                      (Other)    :233                                          
    ##    family_status  phone_operator prev_client bad_client
    ##  Another  :1201   0:536          No : 681    No :1527  
    ##  Married  : 444   1:666          Yes:1042    Yes: 196  
    ##  Unmarried:  78   2:317                                
    ##                   3:177                                
    ##                   4: 27                                
    ##                                                        
    ## 

## Exploratory analysis

### Numeric features

![](Report_files/figure-gfm/EDANumerics-1.png)<!-- -->

### Categorical features

![](Report_files/figure-gfm/EDACat1-1.png)<!-- -->

![](Report_files/figure-gfm/EDACat2-1.png)<!-- -->

## Feature engineering

### Creating new features

``` r
#payment: amount to be paid in 1 payment period
df <- df %>% mutate(payment = round(credit_amount / credit_term, 0), .after=credit_term)

#ratio_amount: total amount to be paid / income
df <- df %>% mutate(ratio_amount = round(credit_amount / income, 2), .after=credit_amount)

#ratio_payment: one payment / income
df <- df %>% mutate(ratio_payment = round(payment / income, 2), .after=payment)
```

### Evaluating new features

![](Report_files/figure-gfm/FeatEngEval-1.png)<!-- -->

## Modeling

### Preprocessing

``` r
#function to perform ordinal encoding on a column
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL)) - 1
  x
}

df$education <- encode_ordinal(df$education, order=levels(df$education))
```

``` r
#create preprocessing pipeline

#yeo-johnson transformation
pipe_yeo <- po("yeojohnson", standardize=FALSE)


#weight of evidence encoding
encode_woe <- po("encodeimpact")


#centering and scaling
pipe_center_scale <- po("scale")


#adding interaction terms with model matrix
pipe_interact <- po("modelmatrix", formula = ~ . + credit_amount:product_type.No +
                      credit_amount:product_type.Yes +
                      age:income + age:family_status.No + age:family_status.Yes +
                      age:have_children.No + age:have_children.Yes)


#adding minority class weights
pipe_weights <- po("classweights")
pipe_weights$param_set$values$minor_weight <- summary(df$bad_client)[1] /
  summary(df$bad_client)[2]


#create combined preprocessing pipeline (called Graph in mlr3)
graph_preproc <- pipe_yeo %>>% encode_woe %>>% pipe_center_scale %>>%
  pipe_interact %>>% pipe_weights
```

``` r
#create classification task
task_credit <- as_task_classif(df, target="bad_client", positive="Yes")


#create 5-folds resampling
resample_cv <- rsmp("cv", folds=5L)


#create performance measures
source("mlr3CustomMeasures.R") #source custom implementation of Cohen's kappa measure
measure_rec <- msr("classif.recall") 
measure_prauc <- msr("classif.prauc") 
measure_brier <- msr("classif.bbrier") 
measure_kappa <- msr("classif.kappa") 
measure_acc <- msr("classif.acc") 
measure_time <- msr("time_both")
measures_list <- c(measure_rec, measure_prauc, measure_brier, measure_kappa, 
                   measure_acc, measure_time)


#create logloss measure, just as a tuning metric
measure_log <- msr("classif.logloss")


#create tuners
tune_grid <- mlr3tuning::tnr("grid_search")
tune_random <- mlr3tuning::tnr("random_search") 
tune_hyper2 <- mlr3tuning::tnr("hyperband", eta=2) 
tune_hyper3 <- mlr3tuning::tnr("hyperband", eta=3) 
tune_anneal <- mlr3tuning::tnr("gensa") 


#create tuning terminators
term10 <- trm("evals", n_evals=10)
term25 <- trm("evals", n_evals=25)
term50 <- trm("evals", n_evals=50)
term100 <- trm("evals", n_evals=100)
term250 <- trm("evals", n_evals=250)
term500 <- trm("evals", n_evals=500)
term_none <- trm("none")
```

### Approach to hyperparameter tuning

``` r
#create preprocessing + model pipeline (GraphLearner in mlr3)
learner_glmnet <- as_learner(graph_preproc %>>% po("learner", 
                                                   learner=lrn("classif.glmnet",
                                                   predict_type="prob")))


#create a search space for tuning
space_glmnet <- ps(
  classif.glmnet.alpha=p_dbl(lower=0, upper=1),
  classif.glmnet.lambda=p_dbl(lower=0, upper=1)
)


#create autotuner object
autotune_glmnet = AutoTuner$new(
  learner=learner_glmnet,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term100,
  tuner=tune_anneal,
  search_space=space_glmnet
)


#train autotuner object on the classification task
set.seed(1922)
with_progress(autotune_glmnet$train(task_credit))


#extract tuning results archive as data table
archive_glmnet <- as.data.table(autotune_glmnet$archive)
#best tune: alpha 0.1231044 lambda 0.04147188 logloss 0.5871771


#set graph learner parameters to best tune from autotuner
learner_glmnet$param_set$values <- autotune_glmnet$learner$param_set$values
```

``` r
#create transformation functions
k_square <- function(k) {
  k <- 2^k
  k
}

k_power10 <- function(k) {
  k <- 10^k
  k
}


#create tuning space, with trafo and budget arguments
space_svm = ps(
  classif.svm.cost=p_int(1, 6, trafo=k_power10, tags="budget"),
  classif.svm.gamma=p_int(-8, 2, trafo=k_square)
)


#create autotuner object
autotune_svm = AutoTuner$new(
  learner=learner_svm,
  resampling=resample_cv,
  measure=measure_log,
  terminator=term_none,
  tuner=tune_hyper2,
  search_space=space_svm
)
```

### Classifiers, with the best tunes found

#### Naive Bayes

``` r
#Naive Bayes graph learner
learner_bayes <- as_learner(graph_preproc %>>% 
                              po("learner", 
                                 learner=lrn("classif.naive_bayes",
                                 predict_type="prob")))
```

#### glmnet

``` r
#glmnet graph learner
learner_glmnet <- as_learner(graph_preproc %>>% 
                               po("learner", 
                                  learner=lrn("classif.glmnet",
                                              predict_type="prob",
                                              alpha=0.1231044,
                                              lambda=0.04147188)))
```

#### kNN

``` r
#kNN graph learner
learner_knn <- as_learner(graph_preproc %>>% 
                            po("learner", 
                              learner=lrn("classif.kknn",
                              predict_type="prob",
                              kernel="optimal",
                              scale=FALSE,
                              k=129,
                              distance=1.843605)))
```

#### SVM

``` r
#named class weights vector
svm_weight <- c(1, pipe_weights$param_set$values$minor_weight)
names(svm_weight) <- c("No", "Yes")


#SVM graph learner
learner_svm <- as_learner(graph_preproc %>>% po("learner", 
                                                learner=lrn("classif.svm",
                                                            type="C-classification",
                                                            class.weights=svm_weight,
                                                            predict_type="prob",
                                                            kernel="radial",
                                                            scale=FALSE,
                                                            cost=10,
                                                            gamma=0.009167361)))
```

#### XGBoost

``` r
#XGBoost graph learner
learner_xgb <- as_learner(graph_preproc %>>% po("learner", 
                                                learner=lrn("classif.xgboost",
                                                            booster="gbtree",
                                                            predict_type="prob",
                                                            nthread=3,
                                                            nrounds=27,
                                                            eta=0.06444704,
                                                            max_depth=12,
                                                            min_child_weight=20,
                                                            max_delta_step=8,
                                                            gamma=0.73399971,
                                                            lambda=0.9294227,
                                                            alpha=0.14594723)))
```

### Benchmarking

``` r
#create baseline learner
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
```

<table style="font-family: Calibri, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif; display: table; border-collapse: collapse; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3;">
  <thead style="">
    <tr>
      <td colspan="7" style="background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-width: 0; font-weight: normal;" style>Performance metrics of loan client classifiers</td>
    </tr>
    <tr>
      <td colspan="7" style="background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; color: #333333; font-size: 85%; font-weight: initial; padding-top: 0; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; font-weight: normal;" style>Averages of 5-folds crossvalidation. Threshold probability=0.5</td>
    </tr>
  </thead>
  <thead style="border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3;">
    <tr>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Classifier</th>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Recall</th>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">PRAUC</th>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Brier score</th>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Cohen's kappa</th>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Accuracy</th>
      <th style="color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 6px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; text-align: center; border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Time</th>
    </tr>
  </thead>
  <tbody style="border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3;">
    <tr><td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center;">Baseline</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #F55000; color: #FFFFFF;">0.1138</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #49F700; color: #000000;">0.1010</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #6BED00; color: #000000;">0.8862</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #B2C300; color: #000000;">0.642</td></tr>
    <tr><td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center;">NaiveBayes</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #CBA800; color: #000000;">0.4820</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #EA7300; color: #000000;">0.2351</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #63F000; color: #000000;">0.1910</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #F06300; color: #000000;">0.1753</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #98D600; color: #000000;">0.7475</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #CAA900; color: #000000;">0.886</td></tr>
    <tr><td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center;">glmnet</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #AFC600; color: #000000;">0.6484</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #E97400; color: #000000;">0.2393</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #66EF00; color: #000000;">0.2023</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #F06300; color: #000000;">0.1755</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #A8CB00; color: #000000;">0.6814</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #B3C100; color: #000000;">0.654</td></tr>
    <tr><td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center;">kNN</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #EB7100; color: #000000;">0.2256</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #47F700; color: #000000;">0.0958</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #6BED00; color: #000000;">0.8862</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FE1A00; color: #FFFFFF;">1.714</td></tr>
    <tr><td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center;">SVM</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #EC6E00; color: #000000;">0.2144</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #48F700; color: #000000;">0.0975</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0300; color: #FFFFFF;">-0.0011</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #6CED00; color: #000000;">0.8856</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #FF0000; color: #FFFFFF;">1.742</td></tr>
    <tr><td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center;">XGBoost</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #C9AB00; color: #000000;">0.4985</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #E87600; color: #000000;">0.2460</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #5FF100; color: #000000;">0.1745</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #F15F00; color: #FFFFFF;">0.1597</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #9DD300; color: #000000;">0.7278</td>
<td style="padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; text-align: center; background-color: #BEB700; color: #000000;">0.760</td></tr>
  </tbody>
  
  
</table>

![](Report_files/figure-gfm/PRCCurves-1.png)<!-- -->

## Conclusions
