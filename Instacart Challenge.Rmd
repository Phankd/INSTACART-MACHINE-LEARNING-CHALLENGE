---
#title: "Instacart Challenge - KEVIN PHAN CANDIDATE SUBMISSION FOR MACHINE LEARNING ENGINEER "
#output: "R MARKDOWN, WORD KNITTED OUTPUT DOCUMENT, TEXT FILE WITH RAW CODE"
---

```{r}
library(ggplot2)
library(reshape)
library(GGally)
library(ROSE)
library(glmnet)
library(plyr)
library(Metrics)
library(party)
library(MASS)
library(leaps)
library(randomForest)
Master_data_Train <- read.csv("/Users/kevinphan/Desktop/instacart-picking-time-challenge-data/train_trips.csv")
Order_times <- read.csv("/Users/kevinphan/Desktop/instacart-picking-time-challenge-data/order_items.csv")
```


```{r}
#Incorporate aggregated quantity data into our dataset. 
QuantityOrder <-sqldf::sqldf("SELECT SUM(quantity) AS TOTAL_QUANTITY, trip_id FROM Order_times GROUP BY trip_id")
Master_data_Train <- join(Master_data_Train,QuantityOrder, by = "trip_id")
```



```{r}
#DATA CLEANSING PART 1
#We will first create a column called duration to see the difference between Start and End time as that will be our dependent variable.
str(Master_data_Train)
Master_data_Train$shopping_started_at <- as.POSIXct(strptime(Master_data_Train$shopping_started_at,"%Y-%m-%d %H:%M:%S")) #Change our start time variable to datetime
Master_data_Train$shopping_ended_at <- as.POSIXct(strptime(Master_data_Train$shopping_ended_at,"%Y-%m-%d %H:%M:%S"))#Change our end time variable to datetime
Master_data_Train$Duration_In_Sec <-abs(60*as.numeric(difftime(Master_data_Train$shopping_started_at,Master_data_Train$shopping_ended_at))) #in seconds
```




```{r}
#DATA CLEANSING PART 2
#We now check for NAs, Nulls, and Outliers. 
table(is.na(Master_data_Train$Duration_In_Sec)) #Check for NA
is.null(Master_data_Train$Duration_In_Sec) # Check for Null 
fivenum(Master_data_Train$Duration_In_Sec) #Just to see our min max and everything in btwee. 
boxplot(Master_data_Train$Duration_In_Sec, ylab = "Seconds", xlab = "Plot", main = "BoxPlot of Duration in Seconds", colour = "Blue")
#We can see from the box plot that there are quite a bit of outliers that can skew our data. 

hist(Master_data_Train$Duration_In_Sec, xlab = "Duration (Seconds)", main = "Distribution Plot of Duration", breaks = 25, col = "Red")
#This histogram also shows our durations skewed to the right meaning there are outliers.

#Remove outliers

#we use Mahalanobis distance since we are in multivariate space and we need to remove outliers from two columns. 
MD <- mahalanobis(Master_data_Train[,c(7,8)], colMeans(Master_data_Train[,c(7,8)]),cov(Master_data_Train[,c(7,8)]),tol=1e-20) #find Mahalanobis Distance for each point. 
Master_data_Train$MD <- round(MD,3) #round distance to 3
Master_data_Train$Outlier_Mahalanobis <- "No"
Master_data_Train$Outlier_Mahalanobis[Master_data_Train$MD > 1] <- "Yes" #Threshold i did chose was 1, for all points having MD > 1. 
Master_data_Train <- Master_data_Train[Master_data_Train$Outlier_Mahalanobis == "No",]
Master_data_Train$Outlier_Mahalanobis <- NULL
Master_data_Train$MD <- NULL #Discard the column since we no longer need it.
#Approximately 4000 rows removed of outliers.

boxplot(Master_data_Train$Duration_In_Sec, ylab = "Seconds", xlab = "Plot", main = "BoxPlot Outliers Removed", colour = "Purple")
boxplot(Master_data_Train$TOTAL_QUANTITY, ylab = "Quantity", xlab = "Plot", main = "BoxPlot Outliers Removed", colour = "Yellow")
hist(Master_data_Train$TOTAL_QUANTITY, xlab = "Quantity", main = "Distribution Plot of Duration No Outliers", breaks = 25, col = "Yellow")

#We can see that the boxplots are much tidier and our distribution looks more gaussian from our new histogram and our box plot.
```


```{r}
#DATA CLEANSING PART 3
#CHECKING FOR IMBALANCES IN THE DATA (BEST FOR CLASSIFICATIONS)
ggplot(data.frame(Master_data_Train$fulfillment_model),aes(x=Master_data_Train$fulfillment_model)) + geom_bar()
#This frequency plot shows the imbalance we have in our fullfillment models. It is not so much a big problem in a numerical predicting model such as regression but in classification algorithms, it will be a problem. It wouldnt hurt to balance this and so i will do just that.

table(Master_data_Train$fulfillment_model) #Large Discrepancy
Balanced_Train <- ovun.sample(fulfillment_model ~ ., data = Master_data_Train, method = "both",p = 0.5)$data 
#This method utlizies both over and undersampling for each model (model 1 and 2).
table(Balanced_Train$fulfillment_model)

#Changing our fullfillment model column to factor where 0 is model 1 and 1 is model 2. 
Balanced_Train$fulfillment_model <- as.character(Balanced_Train$fulfillment_model)
Balanced_Train$fulfillment_model[Balanced_Train$fulfillment_model == 'model_1'] = '0'
Balanced_Train$fulfillment_model[Balanced_Train$fulfillment_model == 'model_2'] = '1'

#We will use the balanced train dataset from now!
```



```{r}
#DATA CLEANSING PART 4
#We will do a normality check on the data using shapiro-wilks test. 

shapiro.test(Balanced_Train$Duration_In_Sec[1:5000])
shapiro.test(Balanced_Train$TOTAL_QUANTITY[1:5000])
#We ue the first 5000 rows because the shapiro wilks test has a test parameter for up to max 5000 rows. 
#Both tests have p values under 0.05. Therefore, we can reject the null hypothesis of normality. 
```



```{r}
#DATA CLEANSING PART 5
#CHECKING FOR MULTICOLLINEARITY AND CHECKING THE SHAPE OF OUR DATA 
Correlations = cor(Balanced_Train[,c(1,2,4,7,8)]) #Using only numeric variables
corrplot::corrplot(Correlations)
#We see no correlation with duration in sec to other numeric attributes becasue they are just IDs. We do see a correlation between duration and quantities. 
#We see absolutely no multicollinearity between variables. So we do not have to perform an variance inflation factor test.

ggpairs(Balanced_Train[,c(1,2,4,7,8)]) #In comparison to others, duration and quantity are correlated!

plot(Balanced_Train$Duration_In_Sec, Balanced_Train$TOTAL_QUANTITY) #this shows that our data is not linear. We will show that a linear model does not perform the best. 
```



```{r}
#our model
#LINEAR REGRESSION
#Remove non predictor variables
set.seed((123))
Balanced_Train <- Balanced_Train[,-c(5,6)] #We removed the start and end date times since we have duration. 
Balanced_Train$fulfillment_model <- as.factor(Balanced_Train$fulfillment_model)
subsets <- regsubsets(Duration_In_Sec ~., data = Balanced_Train, nvmax = 6) #We see here our best variables for each set of predictors.
summary(subsets)
#WE will go first with a linear model and see how this works. 
set.seed(133)
model <- lm(Duration_In_Sec ~., data = Balanced_Train)
summary(model)
stepwiseModel <- stepAIC(model)
#Stepwise gave us the same model because we have already narrowed down our predictors ot the main most powerful ones. 
#We will now attempt forward and backwards elimination using regular subsets. 

```


```{r}
#LASSSO REGRESSION 
set.seed((123))
lambda <- 10^seq(10, -2, length = 100)
x=model.matrix(Duration_In_Sec~.-1,data=Balanced_Train)  #Since Lasso doestn follow function language, we define it with x and y. 
y=Balanced_Train$Duration_In_Sec #This is the variable at inspection. 
lasso_mod =glmnet(x,y,alpha=1, lambda = lambda)
cv.out = cv.glmnet(x, y, alpha = 1) #fitting laso on training.
plot(cv.out) #HEre, we draw a plot of mean squared error to levels of lambda. Lambda is our penalty coefficient. We see that there are large values of MSE for higher degrees of log lambda. 
bestlam = cv.out$lambda.min #We choose the best lambda to minimize or MSE. 
x_test = model.matrix(Duration_In_Sec~.-1, Balanced_Train) #We use our training set to test our MSE again. 
lasso_pred = predict(lasso_mod, s = bestlam, newx = x_test)

rmse(lasso_pred,Balanced_Train$Duration_In_Sec)
```


```{r}
#REGRESSION TREE & Random Forest
tree_mod <- ctree(Duration_In_Sec ~ ., data=Balanced_Train) #Our regression tree model. 
print(tree_mod)
plot(tree_mod, type="simple") #a visual of our tree. 
pred <- predict(tree_mod, Balanced_Train)
rmse(pred, Balanced_Train$Duration_In_Sec)

#We move on to a random forest which does better in my opinion because it allows us to bypass the pruning phase becasue it uses different trees in an ensemble method.
RF_Model <- randomForest(Duration_In_Sec ~ ., data=Balanced_Train)
plot(RF_Model, type = "simple")
pred <- predict(RF_Model, Balanced_Train)
rmse(pred, Balanced_Train$Duration_In_Sec)
#we see that about 100 trees does the best in terms of minimizing error and so we will go with that. 

#We will see for which mtry (number of variables inspected at each split) is optimal. 

Accuracy = vector("numeric",10L)
for (i in seq(1,5,1)) {
  RF_Model <- randomForest(Duration_In_Sec ~ ., data=Balanced_Train[1:20000,], ntree = 100, mtry = i) 
  Pred <- predict(RF_Model,Balanced_Train)
  Accuracy[i] = rmse(pred, Balanced_Train$Duration_In_Sec)
}

Accuracy
#Accuracy stays constant which leads me to believe that there are only a few strong predictors and the rest are noise.
```



```{r}
#OUR CONCLUSION
#We see that the Random Forest, performed the best with the lowest rmse. The reason being is that the dataset is not linear in nature and a random Forest is better at capturing the non-linear effects. The random forest also allows us to bypass the pruning phase and it uses ensemble techniques to piece together the best model. We will now give our predictions.   
```


```{r}
#TEST SET PREDICTIONS 
#import the test set, format it to have the same class and dimmensions as our test set that the random forest model is trained on and predict the duration using our random forest model. We then round those values and take the desired columns for the output. 
TESTSet <- read.csv("/Users/kevinphan/Desktop/instacart-picking-time-challenge-data/test_trips.csv")
TESTSet <- join(TESTSet, QuantityOrder, by = "trip_id")
TESTSet$fulfillment_model <- as.character(TESTSet$fulfillment_model)
TESTSet$fulfillment_model[TESTSet$fulfillment_model == 'model_1'] = '0'
TESTSet$fulfillment_model[TESTSet$fulfillment_model == 'model_2'] = '1'
TESTSet$fulfillment_model <- as.factor(TESTSet$fulfillment_model)

RF_Model <- randomForest(Duration_In_Sec ~ ., data=Balanced_Train, ntree = 100)

TESTSet$shopping_time <- abs(predict(RF_Model, TESTSet))
TESTSet <- TESTSet[,c(1,7)]
TESTSet$shopping_time <- round(TESTSet$shopping_time,0)

TESTPREDICTIONS <- write.csv(TESTSet, "TESTPREDICTIONS.csv")
```



```{r}
#CONSIDERATIONS AND ALTERNATIVES / NEXT STEPS

#We saw that although there were no shortages of rows, we saw the independent variables did not display much of a predicting power. The information gain was greatest with the aggregated quantity associated with each order trip. There was a thought of doing hot one-encoding with the department names but we had 86 unique departments (length(table(Order_times$department_name))) and that would result in a dataset with way too many dimmensions. We could have used principal component in this case but i felt the accrcy would still not be as high as what we have now due to just the sheer size and less then stellar correlations we witnessed. I would say as a suggestions that we include more better predictors such as coupon usage times or product configurations. Therefore, the model is off by about 8 minutes on average. 


```