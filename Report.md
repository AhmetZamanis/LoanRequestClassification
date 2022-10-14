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
set.seed(1923)
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

<div id="avxemykypa" style="overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>@import url("https://fonts.googleapis.com/css2?family=Calibri:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap");
html {
  font-family: Calibri, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
}

#avxemykypa .gt_table {
  display: table;
  border-collapse: collapse;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#avxemykypa .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#avxemykypa .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#avxemykypa .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 0;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#avxemykypa .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#avxemykypa .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#avxemykypa .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#avxemykypa .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#avxemykypa .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#avxemykypa .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#avxemykypa .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#avxemykypa .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
}

#avxemykypa .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#avxemykypa .gt_from_md > :first-child {
  margin-top: 0;
}

#avxemykypa .gt_from_md > :last-child {
  margin-bottom: 0;
}

#avxemykypa .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#avxemykypa .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#avxemykypa .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#avxemykypa .gt_row_group_first td {
  border-top-width: 2px;
}

#avxemykypa .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#avxemykypa .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#avxemykypa .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#avxemykypa .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#avxemykypa .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#avxemykypa .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#avxemykypa .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#avxemykypa .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#avxemykypa .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#avxemykypa .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-left: 4px;
  padding-right: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#avxemykypa .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#avxemykypa .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#avxemykypa .gt_left {
  text-align: left;
}

#avxemykypa .gt_center {
  text-align: center;
}

#avxemykypa .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#avxemykypa .gt_font_normal {
  font-weight: normal;
}

#avxemykypa .gt_font_bold {
  font-weight: bold;
}

#avxemykypa .gt_font_italic {
  font-style: italic;
}

#avxemykypa .gt_super {
  font-size: 65%;
}

#avxemykypa .gt_footnote_marks {
  font-style: italic;
  font-weight: normal;
  font-size: 75%;
  vertical-align: 0.4em;
}

#avxemykypa .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#avxemykypa .gt_indent_1 {
  text-indent: 5px;
}

#avxemykypa .gt_indent_2 {
  text-indent: 10px;
}

#avxemykypa .gt_indent_3 {
  text-indent: 15px;
}

#avxemykypa .gt_indent_4 {
  text-indent: 20px;
}

#avxemykypa .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table">
  <thead class="gt_header">
    <tr>
      <td colspan="7" class="gt_heading gt_title gt_font_normal" style>Performance metrics of loan client classifiers</td>
    </tr>
    <tr>
      <td colspan="7" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style>Averages of 5-folds crossvalidation. Threshold probability=0.5</td>
    </tr>
  </thead>
  <thead class="gt_col_headings">
    <tr>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Classifier</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Recall</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">PRAUC</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Brier score</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Cohen's kappa</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Accuracy</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" style="border-bottom-width: 3px; border-bottom-style: solid; border-bottom-color: #000000; font-weight: bold;" scope="col">Time</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td class="gt_row gt_center">Baseline</td>
<td class="gt_row gt_center" style="background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td class="gt_row gt_center" style="background-color: #F55000; color: #FFFFFF;">0.1137</td>
<td class="gt_row gt_center" style="background-color: #49F700; color: #000000;">0.1009</td>
<td class="gt_row gt_center" style="background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td class="gt_row gt_center" style="background-color: #6BED00; color: #000000;">0.8863</td>
<td class="gt_row gt_center" style="background-color: #B9BC00; color: #000000;">0.718</td></tr>
    <tr><td class="gt_row gt_center">NaiveBayes</td>
<td class="gt_row gt_center" style="background-color: #C6AE00; color: #000000;">0.5171</td>
<td class="gt_row gt_center" style="background-color: #E97600; color: #000000;">0.2451</td>
<td class="gt_row gt_center" style="background-color: #61F100; color: #000000;">0.1815</td>
<td class="gt_row gt_center" style="background-color: #EC6C00; color: #000000;">0.2083</td>
<td class="gt_row gt_center" style="background-color: #95D800; color: #000000;">0.7591</td>
<td class="gt_row gt_center" style="background-color: #C9AA00; color: #000000;">0.888</td></tr>
    <tr><td class="gt_row gt_center">glmnet</td>
<td class="gt_row gt_center" style="background-color: #ADC700; color: #000000;">0.6559</td>
<td class="gt_row gt_center" style="background-color: #E97500; color: #000000;">0.2437</td>
<td class="gt_row gt_center" style="background-color: #65EF00; color: #000000;">0.2004</td>
<td class="gt_row gt_center" style="background-color: #EF6500; color: #000000;">0.1822</td>
<td class="gt_row gt_center" style="background-color: #A7CC00; color: #000000;">0.6854</td>
<td class="gt_row gt_center" style="background-color: #B2C300; color: #000000;">0.650</td></tr>
    <tr><td class="gt_row gt_center">kNN</td>
<td class="gt_row gt_center" style="background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td class="gt_row gt_center" style="background-color: #E87600; color: #000000;">0.2480</td>
<td class="gt_row gt_center" style="background-color: #47F800; color: #000000;">0.0943</td>
<td class="gt_row gt_center" style="background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td class="gt_row gt_center" style="background-color: #6BED00; color: #000000;">0.8863</td>
<td class="gt_row gt_center" style="background-color: #FF0000; color: #FFFFFF;">1.766</td></tr>
    <tr><td class="gt_row gt_center">SVM</td>
<td class="gt_row gt_center" style="background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td class="gt_row gt_center" style="background-color: #EB7000; color: #000000;">0.2231</td>
<td class="gt_row gt_center" style="background-color: #47F800; color: #000000;">0.0954</td>
<td class="gt_row gt_center" style="background-color: #FF0500; color: #FFFFFF;">0.0000</td>
<td class="gt_row gt_center" style="background-color: #6BED00; color: #000000;">0.8863</td>
<td class="gt_row gt_center" style="background-color: #FD2400; color: #FFFFFF;">1.720</td></tr>
    <tr><td class="gt_row gt_center">XGBoost</td>
<td class="gt_row gt_center" style="background-color: #BDB800; color: #000000;">0.5685</td>
<td class="gt_row gt_center" style="background-color: #E87800; color: #000000;">0.2542</td>
<td class="gt_row gt_center" style="background-color: #5EF200; color: #000000;">0.1707</td>
<td class="gt_row gt_center" style="background-color: #EE6900; color: #000000;">0.1954</td>
<td class="gt_row gt_center" style="background-color: #9CD400; color: #000000;">0.7324</td>
<td class="gt_row gt_center" style="background-color: #BABB00; color: #000000;">0.730</td></tr>
  </tbody>
  
  
</table>
</div>

![](Report_files/figure-gfm/PRCCurves-1.png)<!-- -->

## Conclusions
