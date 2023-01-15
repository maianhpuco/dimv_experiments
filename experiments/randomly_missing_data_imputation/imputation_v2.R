## ----setup, include = FALSE-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(cache = TRUE, echo=TRUE, eval = TRUE)


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
require(knitr)
purl("imputation_v2.Rmd", output = 'imputation_v2.R')


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
packages <- c(
  "missMDA", 
  "softImpute", 
  "caret", 
  "caTools", 
  "glue", 
  "jsonlite", 
  "future.apply", 
  "dslabs", 
  "cowplot", 
  "magick", 
  "progress", 
  "datasets", 
  "stats", 
  "foreach", 
  "fdm2id", 
  "datasetsICR", 
  "HDclassif"
)
# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE)) 

library(here) 
source(here('src/rscript/dimv.R'))  
source(here('src/rscript/dpers.R'))   
source(here('src/rscript/utils.R'))    
source(here('src/rscript/imputation_comparation.R'))     

plan(multisession, workers = 8)


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data(iris)
data(ionosphere)
data(seeds) 
data(wine)


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
createRandomlyMissingData = function(data, rate){
  data = as.matrix(data)
  col_num = dim(data)[2] 
  flatten = as.vector(data) 
  
    
  mask = runif(length(flatten), min = 0, max = 1) < rate
  flatten[mask]=NaN
  return(matrix(flatten, ncol = col_num))
}


## ---- message = FALSE, warning = FALSE--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
summary_result <- function(result, caclCol, groupByCol, fold_number, dataset_name, missing_rate, order_decreasing=TRUE){
  result$col = as.numeric(result[, caclCol]) 
  result$imputation  = result[, groupByCol]
  summary = data.frame(
                  group=levels(factor(result$imputation)), 
                  mean=(aggregate(result$col, by=list(result$imputation), FUN=mean)$x),
                  sd=(aggregate(result$col, by=list(result$imputation), FUN=sd)$x), 
                  iteration_times = max(result$fold_number)
             )
  summary = summary[order(summary$mean, decreasing=order_decreasing), ]   
  summary$dataset_name = dataset_name
  summary$missing_rate = missing_rate
  return(summary)
} 


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
summary_all_result <- function(result, acc_col="accuracy", mse_train_col="rmse_train", mse_test_col="rmse_test", groupByCol="imputation_method", iteration="fold_number"){
  result$accuracy = as.numeric(result[, acc_col]) 
  result$rmse_train = as.numeric(result[, mse_train_col])
  result$rmse_test = as.numeric(result[, mse_test_col])
  
  result$imputation  = result[, groupByCol]
  
  summary = data.frame(
        group=levels(factor(result$imputation)), 
        
        accuracy_mean=(aggregate(result$accuracy, by=list(result$imputation), FUN=mean)$x),
        accuracy_sd=(aggregate(result$accuracy, by=list(result$imputation), FUN=sd)$x),  
        
        rmse_train_mean=(aggregate(result$rmse_train, by=list(result$imputation), FUN=mean)$x),
        rmse_train_sd=(aggregate(result$rmse_train, by=list(result$imputation), FUN=sd)$x), 
        
        rmse_test_mean=(aggregate(result$rmse_test, by=list(result$imputation), FUN=mean)$x),
        rmse_test_sd=(aggregate(result$rmse_test, by=list(result$imputation), FUN=sd)$x),  
        
        folds = max(result$fold_number)
         )
  
  summary = summary[order(summary$accuracy_mean, decreasing=T),]   
  summary$accuracy_ranking = rank(-summary$accuracy_mean)
  summary$mse_ranking = rank(summary$rmse_test_mean)
  return(summary)
}  



## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
imputeAndPredictionOnEachFold <- function(fold, missing_data, data, labels, folds, dataset_name, DIMVthreshold){ 
  test_filter = unlist(unname(folds[fold])) 
  
  #filter fold's data 
  rmse_calc <- function(ori_data, imputed_rescaled_data,missing_pos_filter){
    nominator = sum((missing_pos_filter * ori_data - missing_pos_filter * imputed_rescaled_data)**2)
    denominator = sum(missing_pos_filter) 
    return(sqrt(nominator/denominator))
  }
  reconstructingNormedMatrix <- function(X_norm, mean, std){
    mult = sweep(X_norm, 2, std, '*')
    reconstrc = sweep(mult, 2, mean, '+')
    return (reconstrc)
  }
  labels = as.factor(labels)
  
  missing.X_train = missing_data[-test_filter, ] 
  missing.X_test = missing_data[test_filter, ]
  y.train = labels[-test_filter]
  y.test = labels[test_filter]  
  
  train_normed = normalizing(x=missing.X_train, Xtrain=missing.X_train)
  missing.X_train_normed = train_normed$X_normed
  missing.X_train_mean = train_normed$mean
  missing.X_train_sd = train_normed$sd 
  
  test_normed = normalizing(x=missing.X_test, Xtrain=missing.X_train)
  missing.X_test_normed = test_normed$X_normed 
  
  
  func_list = list( 
    'impDi_run', 
    'softImpute_run', 
    'mice_run', 
    'imputePCA_run',  
    'kNNimpute_run', 
    'missForest_run'
    )   
    
  for(j in 1:length(func_list)){
    func_name = unlist(strsplit(func_list[[j]], "_run"))[1] 

    tstart = Sys.time() 
    func <- get(func_list[[j]])  
    if (func_name == "impDi"){
      impted = func(missing.X_train_normed , y.train, missing.X_test_normed, y.test, threshold=DIMVthreshold)
    }else{
      impted = suppressWarnings(func(missing.X_train_normed , y.train, missing.X_test_normed, y.test))  
    }
    
    set.seed(1)
# 
#     print(which(rowSums(is.na(missing.X_train_normed))==dim(data)[2]))
#     print(which(rowSums(is.na(missing.X_test_normed))==dim(data)[2]))

    fit.svm = suppressWarnings(train(as.data.frame(impted$train), y.train, method="svmRadial"))
    
  
    pred <- suppressWarnings((predict(fit.svm, as.data.frame(impted$test))))

    pred <- as.factor(pred)
    acc = mean(pred == y.test)
    
    rmse_train = rmse_calc(
      as.matrix(data[-test_filter,]), 
      reconstructingNormedMatrix(impted$train, missing.X_train_mean, missing.X_train_sd), 
      is.na(missing.X_train)*1 
      )
    
    rmse_test = rmse_calc(
      as.matrix(data[test_filter,]), 
      reconstructingNormedMatrix(impted$test, missing.X_train_mean, missing.X_train_sd),
      is.na(missing.X_test)*1 
      )
     
    
    result = data.frame( 
              list("dataset" = dataset_name, 
                    "fold_number" = fold, 
                    "imputation_method" =  func_name,
                    "accuracy" = acc,
                    "rmse_train" = rmse_train, 
                    "rmse_test" = rmse_test
              ) ) 
    
    if (j > 1){
        results = rbind(results, result)
    }else{
        results = result 
    }
  }
  return(results)
}


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
imputeAndClassificationPipeline <- function(
    dataset, 
    dataset_name, 
    label_col, 
    DIMVthreshold, 
    root, 
    folder_name, 
    missing_rate, 
    number_of_folds){  
 
  print(dataset_name)
  data = dataset[, !names(dataset) %in% c(label_col)]  
  labels_origin = dataset[, label_col,drop=F ]
  
   #shuffle  
  shuffled_idx = sample(1:nrow(data))  
  data = data[shuffled_idx, ] 
  labels = as.numeric(factor(labels_origin[shuffled_idx, ]))   
  

  folds = createFolds(labels, k=number_of_folds)
  missing_data = createRandomlyMissingData(data, missing_rate)  
  pb <- txtProgressBar(min = 0, max = number_of_folds, style = 3) 
  
  # results <- foreach::foreach(i = 1:number_of_folds, .combine='rbind') %dopar% {
  #   setTxtProgressBar(pb, i)  
  #   imputeAndPredictionOnEachFold(i, missing_data, data, labels, folds, dataset_name, DIMVthreshold)
  # }
  for (i in 1:number_of_folds){
    result = imputeAndPredictionOnEachFold(i, missing_data, data, labels, folds, dataset_name, DIMVthreshold) 
    setTxtProgressBar(pb, i) 
    if (i == 1){
      results  = result
    }else{
      results = rbind(results, result)
    }
  } 
  
  results = data.frame(results)
  acc_summary = summary_result(results, 'accuracy', 'imputation_method', "fold_number", dataset_name, missing_rate, order_decreasing=T)  
  rmse_test_summary = summary_result(results, 'rmse_test', 'imputation_method', "fold_number", dataset_name, missing_rate, order_decreasing=F)  
  rmse_train_summary = summary_result(results, 'rmse_train', 'imputation_method', "fold_number", dataset_name,missing_rate,  order_decreasing=F)  
  
  prediction_results = list("acc_summary" = acc_summary, "rmse_summary" = rmse_test_summary)
  
  
  curr_dir = getwd()
  if (dir.exists(file.path(root, dataset_name)) == F){
    dir.create(file.path(root, dataset_name))
  }
  
  if (dir.exists(file.path(root, dataset_name, folder_name))==F){
      dir.create(file.path(root, dataset_name, folder_name))
  }
   
  #path to save result 
  fold_results_dir = file.path(root, dataset_name, folder_name, "fold_results.csv") 
  summary_acc_dir = file.path(root, dataset_name, folder_name, 'acc_summary.csv')
  summary_rmse_test_dir = file.path(root, dataset_name, folder_name, 'rmse_test_summary.csv')
  summary_rmse_train_dir = file.path(root, dataset_name, folder_name, 'rmse_train_summary.csv') 

  write.csv(results, fold_results_dir)
  write.csv(acc_summary, summary_acc_dir) 
  write.csv(rmse_test_summary, summary_rmse_test_dir)
  write.csv(rmse_train_summary, summary_rmse_train_dir)
  print("done saving result")
  return(prediction_results)
} 


## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

executeImputeAndClassification <- function(root, DIMV_THRESHOLD, MISSING_RATE, NUM_FOLDS){


  formatFloat2String <- function(float_value){
    i = as.integer(float_value*100) 
    s = if (as.integer(i/10) < 1){paste0("0", toString(i))}else{toString(i)} 
    return(s)
  } 
  
  folder_name = paste0(
    "missing_rate_",
    formatFloat2String(MISSING_RATE), 
    "_threshold_", 
    formatFloat2String(DIMV_THRESHOLD)
    )   
  print(folder_name)
  
  iris_result = imputeAndClassificationPipeline(
    iris,
   	"iris",
   	"Species",
   	DIMV_THRESHOLD,
   	root,
   	folder_name,
   	MISSING_RATE,
   	NUM_FOLDS)
  
  ionosphere_result = imputeAndClassificationPipeline(
    ionosphere,
   	"ionosphere",
   	"V35",
   	DIMV_THRESHOLD,
   	root,
   	folder_name,
   	MISSING_RATE,
   	NUM_FOLDS)

  seeds_result = imputeAndClassificationPipeline(
    seeds,
   	"seeds",
   	"variety",
   	DIMV_THRESHOLD,
   	root,
   	folder_name,
   	MISSING_RATE,
   	NUM_FOLDS)

  wine_result = imputeAndClassificationPipeline(
    wine,
   	"wine",
   	"class",
   	DIMV_THRESHOLD,
   	root,
   	folder_name,
   	MISSING_RATE,
   	NUM_FOLDS)



  all_dataset_acc_summary = rbind(
    iris_result$acc_summary,
    ionosphere_result$acc_summary,
    seeds_result$acc_summary,
    wine_result$acc_summary
    )

  all_dataset_rmse_summary = rbind(
    iris_result$mse_summary,
    ionosphere_result$rmse_summary,
    seeds_result$rmse_summary,
    wine_result$rmse_summary
    )

  curr_dir = getwd()
  if (dir.exists(file.path(root, folder_name)) == F){
    dir.create(file.path(root, folder_name))
  }
   
  
  acc_dir = file.path(root, folder_name, "accuracy.csv") 
  rmse_dir = file.path(root, folder_name, "rmse.csv")
  
  
  write.csv(all_dataset_acc_summary, acc_dir)
  write.csv(all_dataset_rmse_summary, rmse_dir)
} 



## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

root = "../../data/randomly_missing_dataset/svmRadial_20230113_1"   

NUM_FOLDS = 10

DIMV_THRESHOLD_LIST = c(.3) 
MISSING_RATE_LIST = c(.4,.3,.2,.1)
total = length(DIMV_THRESHOLD_LIST) * length(MISSING_RATE_LIST)

print(total)
count = 0 
for (DIMV_THRESHOLD in DIMV_THRESHOLD_LIST){
  for (MISSING_RATE in MISSING_RATE_LIST){
    print(">>>>>>>>>>>>>>START--------------")
    print(paste0("MISSING_RATE_", MISSING_RATE, "_THRESHOLD_", DIMV_THRESHOLD , "_NUM_FOLDS_", NUM_FOLDS))
    repeat {
    tmp<-try(
      executeImputeAndClassification(root, DIMV_THRESHOLD, MISSING_RATE, NUM_FOLDS)
    )
    if (!(inherits(tmp,"try-error")))
      break
      }

    
    count = count+1 
    print(paste(count, "/", total, " done"))
  }
}





