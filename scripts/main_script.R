########################## START SCRIPT ###########################

# ACKNOWLEDGEMENT:
# In this script, let us
# (1) create a toy data set, and
# (2) run Yin's packages, i.e. YinsLibrary
# COPYRIGHT @ Yiqiao Yin

# README:
# Open a new clean RStudio.
# Open a new R script by pressing ctrl + shift + n
# For easy navigation, please press ctrl + shift + o which will open the menu bar.
# Feel free to copy this script to yours and you can the codes for fast execution.

# SOURCE:
# Data is collected by Department of Transportation and we are responsible for 
# investigating the features that may explain fatalities in NYC.

#################### CREATE ARTIFICIAL DATA SET: CLASSIFICATION ####################

# Set seed
set.seed(2019)

# Create data
n <- 100 # Number of observation
p <- 104 # Number of parameters

# Explanatory variable:
x <- data.frame(matrix(rbinom(n*p,1,1/2),n,p))

# Response variable:
# Please feel free to choose one:
#y <- (x$X1 + x$X2) %% 2
#y <- x$X1^2 + x$X2^2 %% 1
#y <- x$X1 * x$X2 %% 2
#y <- ifelse(exp(x$X1 * x$X2) %% 1 > .5, 1, 0)
#y <- ifelse(sin(x$X1 * x$X2) %% 1 > .5, 1, 0)
#y <- (x$X1 + x$X2 + x$X3) %% 2
#y <- (x$X1 * x$X2 + x$X3 * x$X4) %% 2
#y <- (sin(x$X1 * x$X2) + cos(x$X3 * x$X4)) %% 1; y <- ifelse(y > .5, 1, 0)
#y <- ifelse(rbinom(n,1,1/2), (x$X1+x$X2) %% 2, (x$X3+x$X4+x$X5) %% 2)
y <- ifelse(rbinom(n,1,8/9), (x$X1+x$X2) %% 2,
            ifelse(rbinom(n,1,1/2), (x$X3+x$X4+x$X5) %% 2,
                   ifelse(rbinom(n,1,1/2), (x$X5*x$X6) %% 2,
                          ifelse(rbinom(n,1,1/2), (x$X6+x$X7+x$X8) %% 2,
                                 (x$X3*x$X4+x$X5*x$X10) %% 2))))

# Data frame:
df <- data.frame(cbind(y,x)); all = df; all[1:5,1:3]; dim(all)

# Shuffle:
set.seed(1)
all <- all[sample(1:nrow(all), nrow(all)), ]
all[1:5,1:3]; dim(all)

#################### CREATE ARTIFICIAL DATA SET: REGRESSION ####################

# Set seed
set.seed(2019)

# Create data
n <- 500 # Number of observation
p <- 20 # Number of parameters

# Explanatory variable:
x <- data.frame(matrix(rnorm(n*p,0,1),n,p))
x <- data.frame(matrix(rpois(n*p, lambda = 3),n,p))

# Response variable:
# Please feel free to choose one:
y <- ifelse(((x$X1 + x$X2) > mean(x$X1 + x$X2)), 1, 0)
y <- ifelse(
  rbinom(n,1,1/2),
  as.numeric(((x$X1 + x$X2) > mean(x$X1 + x$X2))),
  ifelse(
    rbinom(n,1,1/2),
    as.numeric(((x$X1 * x$X2) > mean(x$X1 * x$X2))),
    as.numeric(((x$X1 + x$X2 + x$X3) > mean(x$X1 + x$X2 + x$X3)))
  )
)

# Data frame:
df <- data.frame(cbind(y,x)); all = df; all[1:5,1:3]; dim(all)

# Shuffle:
set.seed(1)
all <- all[sample(1:nrow(all), nrow(all)), ]
all[1:5,1:3]; dim(all)

###################### REAL DATA ######################

# Set Working Directory
path <- "C:/Users/eagle/OneDrive/STATS GR8201 - Theo Stat [2019 Spring]/3. VisionZero (Jun 2019)"
setwd(paste0(path, "/data/"))

# Compile Data
all <- read.csv("data.csv")[, -1]
all <- data.frame(apply(all, 2, function(x) as.numeric(x == 0))); raw_data <- all
all[1:2, 1:3]; dim(all)

#################### EXPLORATORY DATA ANALYSIS ########################

# Corrplot
M <- cor(all)
corrplot::corrplot(M, method = "ellipse", tl.cex = 0.2)

# Histogram
psych::multi.hist(all[,1:9])

###################### RUN CLASSIFICATION: #######################

# Performance Comparison
# Grad Descent
Begin.Time <- Sys.time()
GD_Result <- YinsLibrary::Gradient_Descent_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff, alpha = 1e-6, num_iters = 1e5)
End.Time <- Sys.time(); print(End.Time - Begin.Time)
GD_Result$Summary; GD_Result$Prediction.Table; GD_Result$Testing.Accuracy

# Bagging
Begin.Time <- Sys.time()
Bag_Result <- YinsLibrary::Bagging_Classifier(
  x = all[, -1], 
  y = all[, 1], 
  cutoff = cutoff,
  nbagg = 50,
  cutoff.coefficient = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
Bag_Result$Summary; Bag_Result$Prediction.Table; Bag_Result$Testing.Accuracy; Bag_Result$AUC

# GBM
Begin.Time <- Sys.time()
GBM_Result <- YinsLibrary::Gradient_Boosting_Machine_Classifier(
  x = all[, -1],
  y = all[, 1],
  cutoff = cutoff,
  cutoff.coefficient = 1,
  num.of.trees = 100,
  bag.fraction = 0.5,
  shrinkage = 0.05,
  interaction.depth = 3,
  cv.folds = 5)
End.Time <- Sys.time(); print(End.Time - Begin.Time)
GBM_Result$Summary; GBM_Result$Test.Confusion.Matrix; GBM_Result$Test.AUC

# NB
Begin.Time <- Sys.time()
NB_Result <- YinsLibrary::Naive_Bayes(x = all[, -1], y = all[, 1], cutoff = cutoff)
End.Time <- Sys.time(); print(End.Time - Begin.Time)
NB_Result$Summary; NB_Result$Test.Confusion.Matrix; NB_Result$Test.AUC

# Logistic Classifier
Begin.Time <- Sys.time()
Logit_Result <- YinsLibrary::Logistic_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff, fam = gaussian)
End.Time <- Sys.time(); print(End.Time - Begin.Time)
Logit_Result$Summary; Logit_Result$Training.AUC; Logit_Result$Prediction.Table; Logit_Result$AUC

# Lasso/Ridge Classifier
Begin.Time <- Sys.time()
LassoRidgeResult <- YinsLibrary::Lasso_Ridge_Logistic_Classifier(x = all[, -1], y = all[, 1], alpha = 1, cutoff = cutoff)
End.Time <- Sys.time(); print(End.Time - Begin.Time)
LassoRidgeResult$Summary; LassoRidgeResult$Prediction.Table; LassoRidgeResult$AUC

# RF
Begin.Time <- Sys.time()
RF_Result <- YinsLibrary::Random_Forest(
  x = all[, -1],
  y = all[, 1],
  cutoff = cutoff,
  num.tree = 200,
  num.try = sqrt(ncol(all)),
  cutoff.coefficient = 1,
  SV.cutoff = 1:10
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
RF_Result$Summary; RF_Result$Important.Variables; RF_Result$AUC

# iRF
Begin.Time <- Sys.time()
iRF_Result <- YinsLibrary::iter_Random_Forest_Classifier(
  x = all[, -1], 
  y = all[, 1], 
  cutoff = cutoff,
  num.tree = 100,
  num.iter = 10); End.Time <- Sys.time(); print(End.Time - Begin.Time)
iRF_Result$Summary; iRF_Result$Important.Variables; iRF_Result$AUC

# BART
Begin.Time <- Sys.time()
BART_Result <- YinsLibrary::Bayesian_Additive_Regression_Tree_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff)
End.Time <- Sys.time(); print(End.Time - Begin.Time)
summary(BART_Result$Summary); BART_Result$Important.Variables; BART_Result$AUC

# NN
Begin.Time <- Sys.time()
NN_Result <- YinsLibrary::KerasNN(
  x = all[, -1],
  y = all[, 1],
  cutoff = cutoff,
  validation_split = 1 - cutoff,
  batch_size = 128,
  l1.units = 32,
  l2.units = 16,
  l3.units = 8,
  epochs = 6
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
NN_Result$Confusion.Matrix; NN_Result$Testing.Accuracy

# CNN
Begin.Time <- Sys.time()
CNN_Result <- YinsLibrary::KerasC2NN3(
  x = all[, -1],
  y = all[, 1],
  cutoff = cutoff,
  img_rows = 6,
  img_cols = 10,
  batch_size = 64,
  convl1 = 8,
  convl2 = 8,
  convl1kernel = c(2,2),
  convl2kernel = c(2,2),
  maxpooll1 = c(2,2),
  l1.units = 128,
  l2.units = 64,
  l3.units = 32,
  epochs = 50
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
CNN_Result$Summary; CNN_Result$Confusion.Matrix; CNN_Result$Testing.Accuracy; 
tmp = table(as.numeric(CNN_Result$y_test_hat == 0), CNN_Result$y_test); sum(diag(tmp))/sum(tmp)

# CNN
Begin.Time <- Sys.time()
CNN_Result <- YinsLibrary::KerasC6NN3(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.6,
  img_rows = 8,
  img_cols = 8,
  batch_size = 128,
  convl1 = 8, convl2 = 6,
  convl3 = 6, convl4 = 4,
  convl5 = 4, convl6 = 4,
  convl1kernel = c(2,2),
  convl2kernel = c(2,2),
  maxpooll1 = c(2,2),
  l1.units = 256,
  l2.units = 128,
  l3.units = 64,
  epochs = 12
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
CNN_Result$Summary; CNN_Result$Confusion.Matrix; CNN_Result$Testing.Accuracy

# DenseNet
Begin.Time <- Sys.time()
DenseNet_Result <- YinsLibrary::KerasDenseNet(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  img_rows = 8,
  img_cols = 8,
  depth = 40,
  nb_dense_block = 3,
  nb_filter = 16,
  batch_size = 32,
  epochs = 6
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
DenseNet_Result$Summary; DenseNet_Result$Confusion.Matrix; DenseNet_Result$Testing.Accuracy

# ISCORE: FEATURES + MODULES => LOGISTIC
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Classifier(
  x = all[, -1],
  y = all[, 1],
  cut_off = 0.9,
  num.initial.set = 8,
  how.many.rounds = 200,
  num.top.rows = 1,
  k = 4,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Summary; ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Prediction.Table;
ISCORE_Result$Testing.AUC

# ISCORE: MODULES => LOGISTIC
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Module_Classifier(
  x = all[, -1],
  y = all[, 1],
  cut_off = 0.9,
  num.initial.set = 8,
  how.many.rounds = 200,
  num.top.rows = 1,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Summary; ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Testing.AUC

# ISCORE: FEATURES + MODULES => NN
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Classifier_NN(
  x = all[, -1],
  y = all[, 1],
  cut_off = 0.9,
  num.initial.set = 8,
  how.many.rounds = 200,
  num.top.rows = 1,
  l1.units = 256,
  l2.units = 128,
  l3.units = 64,
  num.of.epochs = 30,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Confusion.Matrix; ISCORE_Result$Testing.Accuracy
ISCORE_Result$x_train[1:5, ]

# ISCORE: MODULES => NN
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Module_Classifier_NN(
  x = all[, -1],
  y = all[, 1],
  cut_off = 0.9,
  num.initial.set = 8,
  how.many.rounds = 200,
  num.top.rows = 10,
  l1.units = 256,
  l2.units = 128,
  l3.units = 64,
  num.of.epochs = 30,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Confusion.Matrix; ISCORE_Result$Testing.Accuracy
ISCORE_Result$x_train[1:5, ]

###################### RUN INTERACTION-BASED AI CLASSIFIER ######################

# Check Levels
levels <- c()
for (j in 1:ncol(all)) {levels <- c(levels, nrow(plyr::count(all[,j])))}
levels <- data.frame(cbind(colnames(all), levels))
colnames(levels) <- c("Var_Names","Num_of_Levels"); levels

# ISCORE ONLY
# What to do for more than three levels?
# Convert all X's into two or three levels according to ISCORE
tmp <- YinsLibrary::convert_data_by_iscore(
  x = all[, -1],
  y = all[, 1])
tmp$x_levels
all <- data.frame(cbind(tmp$y, tmp$x_new)); all[1:5, 1:5]; dim(all)

# PREMISE: Maximum number of levels for X's: 0, 1, 2
# Setting up parameters:
num.initial.set = 8
how.many.rounds = 2000
Begin.Time <- Sys.time()
ISCORE <- YinsLibrary::Discrete_VS(
  all = all[, 1:30],
  cut_off = cutoff,
  num.initial.set = num.initial.set,
  how.many.rounds = how.many.rounds,
  num.top.rows = 50,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
head(ISCORE$All.BDA.Modules); head(ISCORE$Top.BDA.Modules.with.Two.or.More.Var)
write.csv(
  ISCORE$All.BDA.Modules[1:30, ],
  paste0(path, "/results/selected_variables_draw_", num.initial.set, "_rounds_", how.many.rounds, "_from_BDA.csv"))

# NAME OF SELECTED VARIABLES BY ISCORE
# What are the names of the variables?
# The following code goes into the names of the dataset and extract those that match ISCORE results.
selected_variables <- data.frame(ISCORE$All.BDA.Modules)[1:10, ]
selected_variables$Variables_Names <- NA; selected_variables
for (i in 1:nrow(selected_variables)) {
  selected_variables[i, 3] <- paste0(colnames(all[,-1])[c(as.numeric(unlist(strsplit(as.character(selected_variables[i, 1]), split="_"))))], collapse = "__")
}; selected_variables[, 2] <- round(as.numeric(as.character(selected_variables[, 2])),3)
colnames(selected_variables) <- c("Top_Variable_Sets", "Importance_Measure", "Variable_Names"); selected_variables
write.csv(
  selected_variables,
  paste0(path, "/results/selected_variables_names_", num.initial.set, "_rounds_", how.many.rounds, "_from_BDA.csv"))

# Visualization:
# Let us plot the important variable sets according to measures.
ggplot2::ggplot(
  selected_variables, 
  aes(Importance_Measure, reorder(Variable_Names, Importance_Measure))) + geom_point() +
  ggtitle("Selected Important Variables in Order of \n Importance Measure") +
  xlab("Importance Measure") + ylab("Variable Names")

# Alternatively, we directly run continuous ISCORE
# PS: I would use discrete version almost always.
# Because continuous require much more tuning.
# Continuous ISCORE
Begin.Time <- Sys.time()
ISCORE <- YinsLibrary::Continuous_VS(
  all = all,
  cut_off = 0.9,
  num.initial.set = 8,
  how.many.rounds = 30,
  num.top.rows = 10,
  seed = 1,
  K.means.K = 3
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
head(ISCORE$All.BDA.Modules)

# INTERACTION-BASED FEATURE EXTRACTION & ENGINEER
# This scripts takes ISCORE results from above
# and update data by using interaction-based
# feature extraction and engineering technique.
Begin.Time <- Sys.time()
NewDataResult <- YinsLibrary::Interaction_Based_Feature_Extraction_and_Engineer(
  x = all[, -1],
  x_original = all[, -1],
  y = all[, 1],
  cut_off = cutoff,
  Result = ISCORE,
  num.top.rows = 3,
  do.you.want.only.modules = FALSE,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
all <- data.frame(cbind(Y = NewDataResult$Y, NewDataResult$X)); head(NewDataResult$Important.Variables)
cbind(colnames(all)); all[1:3, ]; dim(all); all_backup <- all

###################### POST BDA ######################

# ISCORE: FEATURES + MODULES => LOGISTIC
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Adaboost_Classifier_Post_BDA(
  x = all[, -1],
  y = all[, 1],
  cut_off = 200/(200+1e3),
  Result = ISCORE,
  num.top.rows = 1,
  tree_depth = 2,
  n_rounds = 10,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Summary; ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Prediction.Table;
ISCORE_Result$Testing.AUC

# ISCORE: FEATURES + MODULES => LOGISTIC
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Classifier_Post_BDA(
  x = all[, -1],
  y = all[, 1],
  cut_off = 0.9,
  Result = ISCORE,
  num.top.rows = 2,
  k = 3,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Summary; ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Prediction.Table;
ISCORE_Result$Testing.AUC

# ISCORE: FEATURES + MODULES => NN
Begin.Time <- Sys.time()
ISCORE_Result <- YinsLibrary::Interaction_Based_Classifier_NN_Post_BDA(
  x = all[, -1],
  y = all[, 1],
  cut_off = 0.9,
  Result = ISCORE,
  num.top.rows = 3,
  l1.units = 256,
  l2.units = 128,
  l3.units = 64,
  num.of.epochs = 50,
  seed = 1
); End.Time <- Sys.time(); print(End.Time - Begin.Time)
ISCORE_Result$Important.Variables[1:3, ]; ISCORE_Result$Confusion.Matrix; ISCORE_Result$Testing.Accuracy
ISCORE_Result$x_train[1:5, ]

###################### RUN REGRESSION: #######################

# Linear Regression Predictor
Begin.Time <- Sys.time()
LR_Result <- YinsLibrary::Linear_Regression_Predictor(x = all[, -1], y = all[, 1])
End.Time <- Sys.time(); print(End.Time - Begin.Time)
LR_Result$Summary; head(LR_Result$Test); LR_Result$Train.MSE; LR_Result$Test.MSE

# BART Predictor
Begin.Time <- Sys.time()
BART_Result <- YinsLibrary::Bayesian_Additive_Regression_Tree_Predictor(x = all[, -1], y = all[, 1])
End.Time <- Sys.time(); print(End.Time - Begin.Time)
summary(BART_Result$Summary); head(BART_Result$Test); BART_Result$Train.MSE; BART_Result$Test.MSE

# GBM Predictor
Begin.Time <- Sys.time()
GBM_Result <- YinsLibrary::Gradient_Boosting_Machine_Predictor(x = all[, -1], y = all[, 1])
End.Time <- Sys.time(); print(End.Time - Begin.Time)
summary(GBM_Result$Summary); head(GBM_Result$Test); GBM_Result$Train.MSE; GBM_Result$Test.MSE

# RidgeLasso Regression Predictor
Begin.Time <- Sys.time()
RidgeLasso_Result <- YinsLibrary::Lasso_Ridge_Regression_Predictor(x = all[, -1], y = all[, 1])
End.Time <- Sys.time(); print(End.Time - Begin.Time)
summary(RidgeLasso_Result$Summary); head(RidgeLasso_Result$Test); RidgeLasso_Result$Train.MSE; RidgeLasso_Result$Test.MSE

######################### K-FOLD CV: CLASSIFICATION #########################

# READ ME:
# This script loops through k folds.
# Each fold the algorithm fits a selected machine learning technique.
# The algorithm outputs k-fold accuracy (or other selected results). 

# 2 classes:
# This is for classification only
plyr::count(all[,1])
all.A <- all[all[,1] == 0, ]; dim(all.A)
all.B <- all[all[,1] == 1, ]; dim(all.B)

# Null Result:
fold_obs <- c()
Performance <- list()

# Tuning Space
tune_space <- seq(50, 1000, by = 150); tune_space; length(tune_space)

# CV:
# Write a k-fold CV loop:
how.many.folds = 5; folds.i = 1
convert_to_numeric <- FALSE
for (folds.i in 1:(how.many.folds-1)){
  # Create k-fold training data sets for CV:
  
  # Create:
  # folds: a list of numbers with different index;
  # testIndexes: the index that equals to each index in folds;
  
  # For classification
  # Then we can create test and train data sets:
  print(paste0("Fold ", folds.i, " >>> Preparing data ..."))
  folds.A <- cut(seq(1,nrow(all.A)),breaks=how.many.folds,labels=FALSE)
  folds.B <- cut(seq(1,nrow(all.B)),breaks=how.many.folds,labels=FALSE)
  
  # For classifiction: 
  # Set:
  # folds.i <- 1
  testIndexes.A <- which(folds.A==folds.i, arr.ind = TRUE)
  testIndexes.B <- which(folds.B==folds.i, arr.ind = TRUE)
  testData.A <- all.A[testIndexes.A, ]; trainData.A <- all.A[-testIndexes.A, ]
  testData.B <- all.B[testIndexes.B, ]; trainData.B <- all.B[-testIndexes.B, ]
  all <- data.frame(rbind(trainData.A, trainData.B, testData.A, testData.B))
  
  # MODEL FITTING / MACHINE LEARNING:
  # One can change to use Regression or Classification:
  cutoff <- (nrow(all) - (nrow(testData.A) + nrow(testData.B)))/nrow(all); cutoff
  
  # GD
  print(paste0("Training ... using algorithm: Gradient Descent"))
  Begin.Time <- Sys.time()
  GD_Result <- YinsLibrary::Gradient_Descent_Classifier(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    cutoff.coefficient = 1,
    alpha = 1e-06,
    num_iters = 1e3
  ); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  GD_Result$Prediction.Table; GD_Result$Testing.Accuracy
  
  # Bagging
  print(paste0("Training ... using algorithm: Bagging or Bootstrap Aggregation ..."))
  Begin.Time <- Sys.time()
  Bag_Result <- YinsLibrary::Bagging_Classifier(
    x = all[, -1], 
    y = all[, 1], 
    cutoff = cutoff,
    nbagg = tune_space[folds.i],
    cutoff.coefficient = 1
  ); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  Bag_Result$Summary; Bag_Result$Prediction.Table; Bag_Result$Testing.Accuracy; Bag_Result$AUC
  
  # GBM
  print(paste0("Training ... using algorithm: Gradient Bossting Machine or GBM"))
  Begin.Time <- Sys.time()
  GBM_Result <- YinsLibrary::Gradient_Boosting_Machine_Classifier(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    cutoff.coefficient = 1,
    num.of.trees = tune_space[folds.i],
    bag.fraction = 0.5,
    shrinkage = 0.05,
    interaction.depth = 3,
    cv.folds = 5)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  GBM_Result$Summary; GBM_Result$Test.Confusion.Matrix; GBM_Result$Test.AUC
  
  # NB
  print(paste0("Training ... using algorithm: Naive Bayes or NB"))
  Begin.Time <- Sys.time()
  NB_Result <- YinsLibrary::Naive_Bayes_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  NB_Result$Summary; NB_Result$Test.Confusion.Matrix; NB_Result$Test.AUC
  
  # Logistic Classifier
  print(paste0("Training ... using algorithm: Logistic Regression or Genearlized Linear Model choosing Binomial family"))
  if (convert_to_numeric) {for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}}
  Begin.Time <- Sys.time()
  Logit_Result <- YinsLibrary::Logistic_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff, fam = gaussian)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  Logit_Result$Summary; Logit_Result$Training.AUC; Logit_Result$Prediction.Table; Logit_Result$AUC
  
  # RF
  print(paste0("Training ... using algorithm: Random Forest or RF"))
  if (convert_to_numeric) {for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}}
  Begin.Time <- Sys.time()
  RF_Result <- YinsLibrary::Random_Forest_Classifier(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    num.tree = tune_space[folds.i],
    num.try = sqrt(ncol(all)),
    cutoff.coefficient = 1,
    SV.cutoff = 1:4
  ); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  RF_Result$Summary; RF_Result$Important.Variables; RF_Result$AUC
  
  # iRF
  print(paste0("Training ... using algorithm: iterative Random Forest or iRF"))
  if (convert_to_numeric) {for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}}
  Begin.Time <- Sys.time()
  iRF_Result <- YinsLibrary::iter_Random_Forest_Classifier(
    x = all[, -1], 
    y = all[, 1], 
    cutoff = cutoff,
    num.tree = tune_space[folds.i],
    num.iter = 3,
    SV.cutoff = 1:4); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  iRF_Result$Summary; iRF_Result$Important.Variables; iRF_Result$AUC
  
  # BART
  print(paste0("Training ... using algorithm: Bayesian Additive Regression Tree or BART"))
  if (convert_to_numeric) {for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}}
  Begin.Time <- Sys.time()
  BART_Result <- YinsLibrary::Bayesian_Additive_Regression_Tree_Classifier(
    x = all[, -1], 
    y = all[, 1], 
    cutoff = cutoff,
    num.tree = tune_space[folds.i],
    num.cut = 50,
    SV.cutoff = 1:4)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  summary(BART_Result$Summary); BART_Result$Important.Variables; BART_Result$AUC
  
  # NN
  print(paste0("Training ... using algorithm: Artificial Neural Network"))
  for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}
  Begin.Time <- Sys.time()
  NN_Result <- YinsLibrary::KerasNN(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    validation_split = 1 - cutoff,
    batch_size = 128,
    activation = "relu",
    l1.units = 64*folds.i, 
    l2.units = 32,
    l3.units = 16,
    epochs = 30 )
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  NN_Result$Confusion.Matrix; NN_Result$Testing.Accuracy
  
  # Case/Control per Fold
  fold_obs <- cbind(fold_obs, c(folds.i, nrow(trainData.A), nrow(trainData.B), nrow(testData.A), nrow(testData.B)))
  
  # Result
  Performance[[folds.i]] <- data.frame(cbind(
    Algo = c("Gradient Descent (GD)", 
             "Bagging or Bootstrap Aggregation", 
             "Gradient Boosting Machine", 
             "Naive Bayes", 
             "Linear Model or Least Squares", 
             "Random Forest", 
             "iterative Random Forest", 
             "Bayesian Additive Regression Tree (BART)",
             "Deep Artificial Neural Network (DANN)"),
    Result = round(c(GD_Result$Testing.Accuracy,
                     Bag_Result$Testing.Accuracy,
                     GBM_Result$Test.Accuracy,
                     NB_Result$Test.Accuracy,
                     Logit_Result$Testing.Accuracy,
                     RF_Result$Testing.Accuracy,
                     iRF_Result$Testing.Accuracy,
                     BART_Result$Testing.Accuracy,
                     NN_Result$Testing.Accuracy),3) ))
  
  # Finished round
  print(paste("Done with fold", folds.i))
}; beepr::beep(10) # End of CV

# Fold:
fold_obs <- t(fold_obs); colnames(fold_obs) <- c("Fold","#Case_for_Train","#Ctrl_for_Train","#Case_for_Test","#Ctrl_for_Test"); fold_obs
write.csv(
  fold_obs,
  paste0(path, "/results/folds_info_for_", how.many.folds-1, "_fold.csv"))

# Best Parameter:
best_param <- c()
Algo = c("Gradient Descent (GD)", 
         "Bagging or Bootstrap Aggregation", 
         "Gradient Boosting Machine", 
         "Naive Bayes", 
         "Linear Model or Least Squares", 
         "Random Forest", 
         "iterative Random Forest", 
         "Bayesian Additive Regression Tree (BART)",
         "Deep Artificial Neural Network (DANN)"); L <- length(Algo); L
for (which.algo in 1:L) {
  best_param <- c(
    best_param,
    which.max(
      unlist(lapply(c(1:(how.many.folds-1)),  function(i) as.numeric(as.character(Performance[[i]][which.algo, 2])))))) }
BestParam <- data.frame(
  Algo = Algo,
  Best_Fold = best_param ); BestParam

# Save
used_iscore <- "used"
write.csv(
  BestParam,
  paste0(path, "/results/best_parameter_for_", how.many.folds-1, "_fold_",
         used_iscore, "_iscore_top_", ncol(all[,-1]), "_var.csv") )

# Result:
# Performance Comparison:
Result_Mat <- c()
for (i in 1:(how.many.folds-1)) {Result_Mat <- c(Result_Mat, as.numeric(as.character(Performance[[i]]$Result)))}
Final_Performance <- data.frame(cbind(
  Algo = c("Gradient Descent (GD)", 
           "Bagging or Bootstrap Aggregation", 
           "Gradient Boosting Machine", 
           "Naive Bayes", 
           "Linear Model or Least Squares", 
           "Random Forest", 
           "iterative Random Forest", 
           "Bayesian Additive Regression Tree (BART)",
           "Deep Artificial Neural Network (DANN)"),
  CV_Result = round(c(rowMeans(matrix(Result_Mat, nrow=L))),3) )); Performance; Final_Performance

# Save
write.csv(
  Final_Performance,
  paste0(path, "/results/performance_", how.many.folds-1, "_fold_", 
         used_iscore, "_iscore_top_", ncol(all[,-1]), "_var.csv") )

#################### HELD-OUT TEST SET ###########################

# HELD-OUT TEST SET
test_fold_obs <- c()
Test_Performance <- list()
#how.many.folds = 11; folds.i = 1
for (folds.i in how.many.folds){
  # Create k-fold training data sets for CV:
  
  # Create:
  # folds: a list of numbers with different index;
  # testIndexes: the index that equals to each index in folds;
  
  # For classification
  # Then we can create test and train data sets:
  print(paste0("Held-out test set: preparing data ..."))
  folds.A <- cut(seq(1,nrow(all.A)),breaks=how.many.folds,labels=FALSE)
  folds.B <- cut(seq(1,nrow(all.B)),breaks=how.many.folds,labels=FALSE)
  
  # For classifiction: 
  # Set:
  # folds.i <- 1
  testIndexes.A <- which(folds.A==folds.i, arr.ind = TRUE)
  testIndexes.B <- which(folds.B==folds.i, arr.ind = TRUE)
  testData.A <- all.A[testIndexes.A, ]; 
  trainData.A <- all.A[-testIndexes.A, ]
  testData.B <- all.B[testIndexes.B, ]; 
  trainData.B <- all.B[-testIndexes.B, ]
  all.train <- data.frame(rbind(
    trainData.A, trainData.B
  )); all <- data.frame(rbind(
    all.train,
    testData.A
  )); dim(all)
  
  # MODEL FITTING / MACHINE LEARNING:
  # One can change to use Regression or Classification:
  cutoff <- (nrow(all) - (nrow(testData.A) + nrow(testData.B)))/nrow(all); cutoff
  
  # GD
  print(paste0("Training ... using algorithm: Gradient Descent"))
  Begin.Time <- Sys.time()
  GD_Result <- YinsLibrary::Gradient_Descent_Classifier(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    cutoff.coefficient = 1,
    alpha = 1e-06,
    num_iters = 1e3
  ); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  GD_Result$Prediction.Table; GD_Result$Testing.Accuracy
  
  # Bagging
  print(paste0("Training ... using algorithm: Bagging or Bootstrap Aggregation"))
  Begin.Time <- Sys.time()
  Bag_Result <- YinsLibrary::Bagging_Classifier(
    x = all[, -1], 
    y = all[, 1], 
    cutoff = cutoff,
    nbagg = tune_space[BestParam$Best_Fold[[1]]],
    cutoff.coefficient = 1
  ); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  Bag_Result$Summary; Bag_Result$Prediction.Table; Bag_Result$Testing.Accuracy; Bag_Result$AUC
  
  # GBM
  print(paste0("Training ... using algorithm: Gradient Bossting Machine or GBM"))
  Begin.Time <- Sys.time()
  GBM_Result <- YinsLibrary::Gradient_Boosting_Machine_Classifier(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    cutoff.coefficient = 1,
    num.of.trees = tune_space[BestParam$Best_Fold[[2]]],
    bag.fraction = 0.5,
    shrinkage = 0.05,
    interaction.depth = 3,
    cv.folds = 5)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  GBM_Result$Summary; GBM_Result$Test.Confusion.Matrix; GBM_Result$Test.AUC
  
  # NB
  print(paste0("Training ... using algorithm: Naive Bayes or NB"))
  Begin.Time <- Sys.time()
  NB_Result <- YinsLibrary::Naive_Bayes_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  NB_Result$Summary; NB_Result$Test.Confusion.Matrix; NB_Result$Test.AUC
  
  # Logistic Classifier
  print(paste0("Training ... using algorithm: Logistic Regression or Generalized Linear Model choosing Binomial family"))
  Begin.Time <- Sys.time()
  Logit_Result <- YinsLibrary::Logistic_Classifier(x = all[, -1], y = all[, 1], cutoff = cutoff, fam = gaussian)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  Logit_Result$Summary; Logit_Result$Training.AUC; Logit_Result$Prediction.Table; Logit_Result$AUC
  
  # RF
  print(paste0("Training ... using algorithm: Random Forest or RF"))
  for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}
  Begin.Time <- Sys.time()
  RF_Result <- YinsLibrary::Random_Forest_Classifier(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    num.tree = tune_space[BestParam$Best_Fold[[5]]],
    num.try = sqrt(ncol(all)),
    cutoff.coefficient = 1,
    SV.cutoff = 1:4
  ); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  RF_Result$Summary; RF_Result$Important.Variables; RF_Result$AUC
  
  # iRF
  print(paste0("Training ... using algorithm: iterative Random Forest or iRF"))
  for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}
  Begin.Time <- Sys.time()
  iRF_Result <- YinsLibrary::iter_Random_Forest_Classifier(
    x = all[, -1], 
    y = all[, 1], 
    cutoff = cutoff,
    num.tree = tune_space[BestParam$Best_Fold[[6]]],
    num.iter = 20,
    SV.cutoff = 1:4); End.Time <- Sys.time(); print(End.Time - Begin.Time)
  iRF_Result$Summary; iRF_Result$Important.Variables; iRF_Result$AUC
  
  # BART
  print(paste0("Training ... using algorithm: Bayesian Additive Regression Tree or BART"))
  for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}
  Begin.Time <- Sys.time()
  BART_Result <- YinsLibrary::Bayesian_Additive_Regression_Tree_Classifier(
    x = all[, -1], 
    y = all[, 1], 
    cutoff = cutoff,
    num.tree = tune_space[BestParam$Best_Fold[[7]]],
    num.cut = 50,
    SV.cutoff = 1:4)
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  summary(BART_Result$Summary); BART_Result$Important.Variables; BART_Result$AUC
  
  # NN
  print(paste0("Training ... using algorithm: Artificial Neural Network"))
  for (j in 1:ncol(all[,-1])) {all[,-1][,j] <- as.numeric(all[,-1][,j])}
  Begin.Time <- Sys.time()
  NN_Result <- YinsLibrary::KerasNN(
    x = all[, -1],
    y = all[, 1],
    cutoff = cutoff,
    validation_split = 1 - cutoff,
    batch_size = 128,
    activation = "sigmoid",
    l1.units = 64*folds.i, 
    l2.units = 32,
    l3.units = 16,
    epochs = 30 )
  End.Time <- Sys.time(); print(End.Time - Begin.Time)
  NN_Result$Confusion.Matrix; NN_Result$Testing.Accuracy
  
  # Case/Control per Fold
  test_fold_obs <- cbind(test_fold_obs, c(folds.i, nrow(trainData.A), nrow(trainData.B), nrow(testData.A), nrow(testData.B)))
  
  # Result
  Test_Performance[[1]] <- data.frame(cbind(
    Algo = c("Gradient Descent (GD)", 
             "Bagging or Bootstrap Aggregation", 
             "Gradient Boosting Machine", 
             "Naive Bayes", 
             "Linear Model or Least Squares", 
             "Random Forest", 
             "iterative Random Forest", 
             "Bayesian Additive Regression Tree (BART)",
             "Deep Artificial Neural Network (DANN)"),
    Result = round(c(GD_Result$Testing.Accuracy,
                     Bag_Result$AUC, 
                     GBM_Result$Test.AUC,
                     NB_Result$Test.AUC,
                     Logit_Result$AUC,
                     RF_Result$AUC,
                     iRF_Result$AUC,
                     BART_Result$AUC,
                     NN_Result$Testing.Accuracy),3) ))
  
  # Finished round
  print(paste("Done with fold", folds.i))
}; beepr::beep(10) # End of Held-out Test Set

# Collect Result
test_fold_obs <- t(test_fold_obs); colnames(test_fold_obs) <- c("Fold","#Case_for_Train","#Ctrl_for_Train","#Case_for_Test","#Ctrl_for_Test"); test_fold_obs
write.csv(
  test_fold_obs,
  paste0(path, "/results/folds_info_for_", how.many.folds, "_fold_test_fold.csv") )

# Test
Held_out_Test <- data.frame(
  Algo = c("Gradient Descent (GD)", 
           "Bagging or Bootstrap Aggregation", 
           "Gradient Boosting Machine", 
           "Naive Bayes", 
           "Linear Model or Least Squares", 
           "Random Forest", 
           "iterative Random Forest", 
           "Bayesian Additive Regression Tree (BART)",
           "Deep Artificial Neural Network (DANN)"),
  Test_Result = as.numeric(as.character(unlist(Test_Performance[[1]]$Result))) ); Held_out_Test

# Save
# used_iscore = "used"
write.csv(
  Held_out_Test,
  paste0(path, "/results/performance_", how.many.folds, "_fold_test_fold_", 
         used_iscore, "_iscore_top_", ncol(all[,-1]), "_var.csv") )

####################### FINISHING MESSAGE #######################

# This code creates a text message
# and saved it on desktop
# and will upload to RSTUDIO and
# pronouce it.
YinsLibrary::job_finished_warning(
  path = "C:/Users/eagle/Desktop/",
  speed = 2,
  content = c(
    "Mr. Yin, your program is ready.",
    "Sir, your code is finished.",
    "Running program is done, sir.",
    "Mr. Yin, program has just finished running."),
  choose_or_random = TRUE )

######################### END SCRIPT #############################