require("missMethods");

MeanImp_run = function(X_train, y_train, X_test, y_test) {
	# Number of train & test sample
	n_train = nrow(X_train);
	n_test	= nrow(X_test);

	# Combining train & test
	X_bind = rbind(X_train, X_test);
	
	# Impute train data with mean
	X_bind_imp = impute_mean(X_bind);

	# Impute test data with train's mean
	X_train_imp = head(X_bind, n=n_train);
	X_test_imp	= tail(X_bind, n=n_test);

	return(list(
				"train"	: X_train_imp,
				"test"	: X_test_imp
	));
};
