#-----------------------------
#
#  Graduate School Admission Prediction
#
#  Purpose: Show usage of evaluation techniques
#
#  Created by: Nikola Tesla (ntesla@uw.edu)
#
#  Created on: 2015-05-07
#
#-----------------------------
set.seed(42)

##----Load Data-----
data <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")

##----Create Factor from rank-----
data$rank <- factor(data$rank)

##----Split Train/Test----
train_indices = sample(1:nrow(data), size=0.8*nrow(data))
train_data = data[train_indices,]
test_data = data[-train_indices,]
rm(data)

##----Fit Logistic Model-----
mylogit <- glm(admit ~ gre + gpa + rank, data = train_data, family = "binomial")

##----Compute Predictions----
train_predictions = predict(mylogit, train_data, type="response")
test_predictions = predict(mylogit, test_data, type="response")

train_data$pred_probs = train_predictions
test_data$pred_probs = test_predictions

train_data$admit_prediction = (train_data$pred_probs > 0.5) + 0
test_data$admit_prediction = (test_data$pred_probs > 0.5) + 0


##----Create Confusion Matrix----
conf_matrix = table(test_data$admit_prediction, test_data$admit,
                    dnn = c('Predicted', 'Actual'))

##----Create K-S statistic----
x_val = seq(0,1,0.01) # Create generic x values to iterate over (x = threshold)
pos_pred = sapply(x_val, function(x) {
  sum(test_data$admit[test_data$pred_probs<=x]==1)/sum(test_data$admit==1)
})

neg_pred = sapply(x_val, function(x) { # Remember (1-x) to reverse line
  sum(test_data$admit[test_data$pred_probs>=(1-x)]==0)/sum(test_data$admit==0)
})

# Plot output
plot(x_val, pos_pred, type='l', col='blue', xlab='Score',
     ylab='Cumulative % Correct', main='Kolmogorov-Smirnov Separation Statistic')
lines(x_val, neg_pred, type='l', col='red')

k_s_ind = which.max(pos_pred - neg_pred) # Find the index of maximum distance
k_s_stat = max(pos_pred - neg_pred) # Find max distance

lines(c(x_val[k_s_ind], x_val[k_s_ind]), c(neg_pred[k_s_ind], pos_pred[k_s_ind]))
legend('topleft', c('Positive Pred', 'Negative Pred'), col=c('blue','red'), lty=c(1,1))
text(0.7,0.7, paste('K-S: ',round(k_s_stat,3)))


##-----ROC Curve (AUC)----
# We will use the x_val from above for the threshold
false_pos = sapply(x_val, function(x){
  num_false_pos = sum(test_data$pred_probs[test_data$admit==0] >= x)
  tot_false = sum(test_data$admit==0)
  return(num_false_pos/tot_false)
})

true_pos = sapply(x_val, function(x){
  num_false_pos = sum(test_data$pred_probs[test_data$admit==1] >= x)
  tot_false = sum(test_data$admit==1)
  return(num_false_pos/tot_false)
})

# Plot Curve
plot(false_pos, true_pos, type='l', main='AUC: Admission Prediction',
     xlab='False Positive Rate', ylab='True Positive Rate', lwd=2)
abline(0, 1, lwd=2, col='red')

# Calculate AUC
get_auc = function(prob, true_vals){
  prob_sort_ind = order(prob, decreasing=T)
  probs_sorted = prob[prob_sort_ind]
  
  y_sort = true_vals[prob_sort_ind]
  x_stack = cumsum(y_sort == 0)/sum(y_sort == 0) # False Positives
  y_stack = cumsum(y_sort == 1)/sum(y_sort == 1) # True Negatives
  
  auc = sum((x_stack[2:length(y_sort)]-x_stack[1:length(y_sort)-1])*
              y_stack[2:length(y_sort)])
  return(auc)
}

AUC = get_auc(test_data$pred_probs, test_data$admit)
