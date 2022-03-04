# Final year project analysis code
## CONTRIBUTERS
# 1. ALVIN OCHIENG - ochiengalvin465@gmail.com
# 2. MICHELLE NGUNYA
# 3. IRENE JEPNG'ETICH
# 4. CAROLINE OSETE
# Loading the libraries 
library(tidyverse)
library(MASS)
library(car)
library(e1071)
library(caret)
library(cowplot)
library(caTools)
library(pROC)
library(ggcorrplot)
library(readr)
library(xgboost)
library(magrittr)
library(Matrix)
library(randomForest)
library(dplyr)
# Importing the dataset into r 
telco <- read_csv("churn_analysis.csv")
View(telco)
colnames(telco)
# Removing the 'Timestamp', 'name', 'reg number' and 'safaricom user'
# columns from the dataset
telco <- telco[,-c(1,2,4,5,6)]
View(telco)
telco <- telco[complete.cases(telco),]
glimpse(telco)
# Transforming the variables into suitable datatypes for analysis
telco$gender <- as.factor(ifelse(telco$gender == "Female", "Female","Male"))
telco$churn <- as.factor(ifelse(telco$churn == 0, "NO", "YES"))
telco$data_package <- as.factor(ifelse(telco$data_package == "Monthly Package","Monthly",
                                       ifelse(telco$data_package == "Daily package",
                                              "Daily", "Weekly")))
telco$tenure <- as.factor(ifelse(telco$tenure == '0-2', '0-2 yrs',
                                 ifelse(telco$tenure == "4-Feb", '2-4 yrs', '4+ yrs')))
telco$networks_subscribed_to <- as.factor(ifelse(telco$networks_subscribed_to
                                                 == 1, '1',
                                                 ifelse(telco$networks_subscribed_to == 2,'2','3')))
str(telco)
# visualizing overall churn
telco %>% group_by(churn) %>% 
        summarise(Count = n()) %>% 
        mutate(percent = prop.table(Count)*100) %>% 
        ggplot(aes(reorder(churn, -percent), percent), fill = churn)+
        geom_col(fill = c("#FC4E07", "#E7B800")) +
        geom_text(aes(label = sprintf("%.2f%%", percent)), hjust = 0.01,
                  vjust = -0.5, size = 3) +
        theme_bw() +
        xlab("Churn") +
        ylab("Percent") +
        ggtitle("Churn Percent")
# Visualizing the categorical data with respect to churn
plot_grid(ggplot(telco, aes(x = gender, fill = churn)) + geom_bar() + theme_bw(),
          ggplot(telco, aes(x = networks_subscribed_to, fill = churn)) + geom_bar() + theme_bw(),
          ggplot(telco, aes(x = data_package, fill = churn)) + geom_bar() + theme_bw(),
          ggplot(telco, aes(x = tenure, fill = churn)) + geom_bar() + theme_bw()
)
ggplot(telco, aes(y = data_charges, x = " ", fill = churn)) + geom_boxplot() +
        theme_bw() + xlab(" ")
ggplot(telco, aes(y = calls_charges, x = " ", fill = churn)) + geom_boxplot() +
        theme_bw() + xlab(" ")
# Checking the correlation between the continuous variables
telco_cor <- cor(telco[, c("data_charges", "calls_charges")])
telco_cor
# data_charges has a positive correlation with call_charges
ggcorrplot(telco_cor, title = "Correlation") + theme(plot.title = element_text(hjust = 0.5))
# checking for outliers in the continuous variables
boxplot(telco$data_charges)$out
# There are 3 outliers in the data_charges column
# Removing the outliers from the dataset
outliers <- boxplot(telco$data_charges)$out
x <- telco
x <- x[-which(telco$data_charges %in% outliers),]
# Checking that all the outliers in the data_charges column have been removed
boxplot(x$data_charges)$out
# Checking for outliers in the calls_charges column
boxplot(x$calls_charges)$out
# there are 5 outliers in the calls_charges column
# Removing the outliers from the dataset
outliers1 <- boxplot(x$calls_charges)$out
y <- x
y <- y[-which(y$calls_charges %in% outliers1),]
# Checking that all the outliers in the calls_charges column have been removed
boxplot(y$calls_charges)$out
y <- telco
## DATA PREPARATION
# Cleaning the Categorical features
# Standardising Continuous features using the scale() function to center the 
# numeric columns
num_columns <- c("data_charges", "calls_charges")
telco[num_columns] <- sapply(telco[num_columns], as.numeric)
telco_int <- telco[,c("data_charges", "calls_charges")]
telco_int <- data.frame(scale(telco_int))
View(telco_int)
# Creating dummy variables for factor variables
telco_cat <- telco[,c(1,2,3,4,7)]
dummy <- data.frame(sapply(telco, function(x) data.frame(model.matrix(~x-1,
                                                                      data = telco_cat))[,-1]))
View(dummy)
# Creating the final dataset by combining the numeric and dummy dataframes
telco_final <- cbind(telco_int, dummy)
View(telco_final)
# Splitting the data into train and validation set.
set.seed(123)
indices <- sample.split(telco_final$churn, SplitRatio = 0.7)
train <- telco_final[indices,]
validation <- telco_final[!(indices),]
## MODEL BUILDING
# starting with logistic regression
# Building the first model with all variables
model_1 <- glm(churn ~ ., data = train, family = 'binomial')
summary(model_1)
# Using stepAIC for variable selection, which is a iterative process
# of adding or removing variables, in order to get a subset of variables
# that gives the best performing model.
model_2 <- stepAIC(model_1, direction = 'both')
summary(model_2)
# We can use variance inflation factor (vif) to get rid of redundant
# predictors or the variables that have high multicollinearity between them.
# Multicollinearity exists when two or more predictor variables are highly
# related to each other and then it becomes difficult to understand the
# impact of an independent variable on the dependent variable.

# The Variance Inflation Factor(VIF) is used to measure the 
# multicollinearity between predictor variables in a model.
# A predictor having a VIF of 2 or less is generally considered safe
# and it can be assumed that it is not correlated with other predictor
# variables. Higher the VIF, greater is the correlation of the predictor
# variable w.r.t other predictor variables. However, Predictors with 
# high VIF may have high p-value(or highly significant), hence, we need
# to see the significance of the Predictor variable before removing it
# from our model.
vif(model_2)
model_3 <- glm(formula = churn ~ data_charges + calls_charges +
                       tenure.x4..yrs + networks_subscribed_to.x2 + 
                       networks_subscribed_to.x3 + data_package.xWeekly + 
                       tenure.x2.4.yrs + data_package.xMonthly,
               family = 'binomial', data = train)
summary(model_3)
vif(model_3)
final_model <- model_3
final_model
# Model evaluation using the validation data
colnames(validation)
prediction <- predict(final_model, type = 'response', newdata = validation[,-4])
summary(prediction)
validation$prob <- prediction
# Using probability cutoff of 50%
pred_churn1 <- factor(ifelse(prediction >= 0.5, 'Yes', 'No'))
actual_churn1 <- factor(ifelse(validation$churn == 1, "Yes", "No"))
table(pred_churn1,actual_churn1)
# Let's find the Accuracy, Sensitivity, Specificity using 50% cutoff

cutoff_churn2 <- factor(ifelse(predition >=0.7, "Yes", "No"))
conf_fin <- confusionMatrix(cutoff_churn2, actual_churn1, positive = "Yes")
accuracy2 <- conf_fin$overall[1]
sensitivity2 <- conf_fin$byClass[1]
specificity2 <- conf_fin$byClass[2]
accuracy2
sensitivity2
specificity2

varImp(final_model)

# Model building 2 
# RANDOM FOREST
# install.packages("randomForest")
set.seed(123)
# indices <- sample.split(telco_final$churn, SplitRatio = 2/3)
ind <- sample(2, nrow(telco_final), replace = TRUE, prob = c(0.7,0.3))
train <- telco_final[ind == 1,]
test <- telco_final[ind == 2,]
model.rf <- randomForest(churn ~ ., data=train, proximity=TRUE,importance = FALSE,
                         ntree=500,mtry=4, do.trace=FALSE)
print(model.rf)
model_rf <- train(churn ~., tuneLength = 15, data = train, method = "rf",
                  importance = TRUE, 
                  trControl = trainControl(method = "repeatedcv",
                                           number = 10,
                                           savePredictions = "final",
                                           classProbs = T))
summary(model_rf)
testpred <- predict(model.rf, type = 'response', newdata = test[,-4])
testpred <- data.frame(testpred)
cutoff_churn <- factor(ifelse(testpred >= 0.7, "Yes", "No"))
actual_churn <- factor(ifelse(test$churn == 1, "Yes", "No"))
df <- data.frame(actual_churn, cutoff_churn)
table(cutoff_churn, actual_churn)

conf_final <- confusionMatrix(cutoff_churn, actual_churn)
conf_final
accuracy <- conf_final$overall[1]
sensitivity <- conf_final$byClass[1]
specificity <- conf_final$byClass[2]
accuracy
sensitivity
specificity

# Below is the variable importance plot, that shows the most significant
# attribute in decreasing order by mean decrease in Gini. The Mean 
# decrease Gini measures how pure the nodes are at the end of the tree.
# Higher the Gini Index, better is the homogeneity.
# Checking the variable Importance Plot
varImpPlot(model.rf)

##MODEL 3
##XGBOOST ALGORITHM
# install.packages("xgboost")
##xgboost final code
set.seed(123)
indices <- sample.split(telco_final$churn, SplitRatio = 0.7)
train <- telco_final[indices,]
test <- telco_final[!(indices),]
train_label <- train[,'churn']
test_label <- test[,'churn']
bst <- xgboost(data = as.matrix(train[,-4]),
               label = train_label,
               max.depth = 2,
               eta = 2,
               nthread = 2,
               objective = "binary:logistic",
               nrounds = 100,
               verbose = 1)
summary(bst)
pre <- predict(bst, as.matrix(test[,-4]))
pre <- as.numeric(pre > 0.6)
confusion <-confusionMatrix(factor(pre), factor(test_label))
confusion
table(factor(pre), factor(test_label))
accuracy3 <- confusion$overall[1]
sensitivity3 <- confusion$byClass[1]
specificity3 <- confusion$byClass[2]
accuracy3
sensitivity3
specificity3

xgb.importance(model = bst)
























