#MLR3 CUSTOM PERFORMANCE MEASURES

library(mlr3)
library(tidymodels)



#WITH YARDSTICK




#MACRO AVERAGE RECALL FOR MULTICLASS CLASSIFICATION


#mlr class
MeasureMRecallMacro = R6::R6Class("MeasureMRecallMacro",
                                  inherit=mlr3::MeasureClassif,
                                  public=list(
                                    initialize = function() {
                                      super$initialize(
                                        #custom id for the measure
                                        id = "classif.mrecall_macro",
                                        
                                        #additional packages required to calculate measure
                                        packages=c("mlr3measures", "yardstick"),
                                        
                                        #required predict type of the learner
                                        predict_type="response",
                                        
                                        #feasible range of values
                                        range=c(0,1),
                                        
                                        #minimize during tuning?
                                        minimize=FALSE
                                      )
                                    }
                                    
                                  ),
                                  
                                  private = list(
                                    #custom scoring function operating on the prediction object
                                    .score=function(prediction, ...){
                                      mrecall_macro <- function(prediction){
                                        rec <- yardstick::recall(data=as.data.table(prediction), 
                                                                 truth=truth, estimate=response,
                                                                 estimator="macro")
                                        mean(rec$.estimate)
                                      }
                                      
                                      mrecall_macro(prediction)
                                    }
                                  )
)

#add to mlr measures
mlr3::mlr_measures$add("classif.mrecall_macro", MeasureMRecallMacro)









#MACRO WEIGHTED AVERAGE RECALL FOR MULTICLASS CLASSIFICATION


#mlr class
MeasureMRecallWMacro = R6::R6Class("MeasureMRecallWMacro",
                                   inherit=mlr3::MeasureClassif,
                                   public=list(
                                     initialize = function() {
                                       super$initialize(
                                         #custom id for the measure
                                         id = "classif.mrecall_wmacro",
                                         
                                         #additional packages required to calculate measure
                                         packages=c("mlr3measures", "yardstick"),
                                         
                                         #required predict type of the learner
                                         predict_type="response",
                                         
                                         #feasible range of values
                                         range=c(0,1),
                                         
                                         #minimize during tuning?
                                         minimize=FALSE
                                       )
                                     }
                                     
                                   ),
                                  
                                  private = list(
                                    #custom scoring function operating on the prediction object
                                    .score=function(prediction, ...){
                                      mrecall_wmacro <- function(prediction){
                                        rec <- yardstick::recall(data=as.data.table(prediction), 
                                                                 truth=truth, estimate=response,
                                                                 estimator="macro_weighted")
                                        mean(rec$.estimate)
                                      }
                                      
                                      mrecall_wmacro(prediction)
                                    }
                                  )
                                  )

#add to mlr measures
mlr3::mlr_measures$add("classif.mrecall_wmacro", MeasureMRecallWMacro)








#MACRO AREA UNDER PRECISION RECALL CURVE FOR MULTICLASS CLASSIFICATION


#mlr class
MeasureMPrAucMacro = R6::R6Class("MeasureMPrAucMacro",
                                  inherit=mlr3::MeasureClassif,
                                  public=list(
                                    initialize = function() {
                                      super$initialize(
                                        #custom id for the measure
                                        id = "classif.mprauc_macro",
                                        
                                        #additional packages required to calculate measure
                                        packages=c("mlr3measures", "yardstick"),
                                        
                                        #required predict type of the learner
                                        predict_type="prob",
                                        
                                        #feasible range of values
                                        range=c(0,1),
                                        
                                        #minimize during tuning?
                                        minimize=FALSE
                                      )
                                    }
                                    
                                  ),
                                  
                                  private = list(
                                    #custom scoring function operating on the prediction object
                                    .score=function(prediction, ...){
                                      m_prauc_func <- function(prediction){
                                        prediction <- as.data.table(prediction)
                                        m_prauc <- yardstick::pr_auc(data=prediction, 
                                                                 truth=truth, 
                                                                 c(names(prediction)[4:ncol(prediction)]),
                                                                 estimator="macro")
                                        mean(m_prauc$.estimate)
                                      }
                                      
                                      m_prauc_func(prediction)
                                    }
                                  )
)

#add to mlr measures
mlr3::mlr_measures$add("classif.mprauc_macro", MeasureMPrAucMacro)








#MACRO WEIGHTED AREA UNDER PRECISION RECALL CURVE FOR MULTICLASS CLASSIFICATION


#mlr class
MeasureMPrAucWMacro = R6::R6Class("MeasureMWPrAucWMacro",
                            inherit=mlr3::MeasureClassif,
                            public=list(
                              initialize = function() {
                                super$initialize(
                                  #custom id for the measure
                                  id = "classif.mprauc_wmacro",
                                  
                                  #additional packages required to calculate measure
                                  packages=c("mlr3measures", "yardstick"),
                                  
                                  #required predict type of the learner
                                  predict_type="prob",
                                  
                                  #feasible range of values
                                  range=c(0,1),
                                  
                                  #minimize during tuning?
                                  minimize=FALSE
                                )
                              }
                              
                            ),
                            
                            private = list(
                              #custom scoring function operating on the prediction object
                              .score=function(prediction, ...){
                                m_prauc_func <- function(prediction){
                                  prediction <- as.data.table(prediction)
                                  m_prauc <- yardstick::pr_auc(data=prediction, 
                                                               truth=truth, 
                                                               c(names(prediction)[4:ncol(prediction)]),
                                                               estimator="macro_weighted")
                                  mean(m_prauc$.estimate)
                                }
                                
                                m_prauc_func(prediction)
                              }
                            )
)

#add to mlr measures
mlr3::mlr_measures$add("classif.mprauc_wmacro", MeasureMPrAucWMacro)




#COHEN'S KAPPA

#mlr class
MeasureKappa = R6::R6Class("MeasureKappa",
                                  inherit=mlr3::MeasureClassif,
                                  public=list(
                                    initialize = function() {
                                      super$initialize(
                                        #custom id for the measure
                                        id = "classif.kappa",
                                        
                                        #additional packages required to calculate measure
                                        packages=c("mlr3measures", "yardstick"),
                                        
                                        #required predict type of the learner
                                        predict_type="response",
                                        
                                        #feasible range of values
                                        range=c(0,1),
                                        
                                        #minimize during tuning?
                                        minimize=FALSE
                                      )
                                    }
                                    
                                  ),
                                  
                                  private = list(
                                    #custom scoring function operating on the prediction object
                                    .score=function(prediction, ...){
                                      kappa_func <- function(prediction){
                                        kappa_val <- yardstick::kap(data=as.data.table(prediction), 
                                                                 truth=truth, estimate=response)
                                        mean(kappa_val$.estimate)
                                      }
                                      
                                      kappa_func(prediction)
                                    }
                                  )
)

#add to mlr measures
mlr3::mlr_measures$add("classif.kappa", MeasureKappa)







