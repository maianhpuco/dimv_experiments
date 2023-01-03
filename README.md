Version 2: 


Version 1: Notebook Note
# dimv's experiment note 
# clf/v5
- About: Implementation of DIMV on MNIST: using all features in impDi
- Result: impDi have lower acc than softImpute 
- Note: The Covariance Matrix contains many very small value 
# clf/v6.Rmd
- About: Similar to v5 but refractoring code. 
- Result: same as v5 
# clf/v7.Rmd
- About: during the imputation procedure (impDi): instead of using all features (with even low correlation feature), this experiment only use the feature that have correlation larger than a threshold with at least x% set of missing feature. 
- Specify: threshold: 0.1, x% = 30%
- Result: the visualization of number after impute show that the imputation still quite random
# clf/v8.Rmd
- About: impDi - during the imputation procedure: loop via each missing feature, find set of observed features that have correlation with missing feature larger than a threshold, if other missing features have the same pattens then we would impute them at the same time. Then we impute these missing feature as impDi specified. 
- Result: on Classification after impute using SVM (one time run, no cross validation) 
+ DIMV: 0.906
+ softImpute: 0.899 

