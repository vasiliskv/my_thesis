Python code for my thesis

My thesis tries to tackle an outlier detection problem, in the form of identifying samples that are unclean from a mixed dataset. The main goal is to test whether there is an effective machine learning pipeline that can solve the problem with satisfactory results. In order to do that, two approaches are followed:

 - The first one utilizes supervised learning algorithms, and therefore the task is seen as a binary classification problem. For this approach the available dataset was 4k labeled samples, which was split in train and test sets. Four different models were tried out, Random Forest, KNN, SVM and Logistic Regression. The best achieved result was measured with the area under the curve metric and was 0.97.
 - The second one utilizes unsupervised learning algorithms. For this approach there was a dataset of 10k unlabeled samples, that was used for the training of the models, and the aforementioned labeled dataset for their evaluation. Two different models were tried out, Isolation Forest and OneClass SVM. The best achieved result was measured with the area under the curve metric and was 0.83.
 
The pipeline created is as follows:
	
 - Data cleaning and normalization
	 - Dropped the highly correlated features and 0-1 normalized the features
 - Feature Selection (only in supervised)
	 - Best feature list for each model out of a number of feature selection methods  
 - Hyperparameter optimization (only in supervised)
 - Model training
 - Model evaluation
