# A Short Survey in VisionZero Project

<p align="center">
  <img width="1000" src="https://github.com/yiqiao-yin/Investigation-of-High-order-Interactions-in-VisionZero-Project/blob/master/figs/background.gif">
</p>

This project investigates the higher-order interactions in VisionZero Traffic Dataset. The following provides a brief for audience. For more detailed report, please go to [main report](https://github.com/yiqiao-yin/Investigation-of-High-order-Interactions-in-VisionZero-Project/blob/master/docs/main.pdf). At predicting whether there is a fatality on the street given road conditions, we deliver strategies at above 94% while benchmark given was around 80%, a 70% error reduction.

# Introduction

This project explores VisionZero Traffic Dataset, a joint project with Department of Transportation (DOT) in New York City. We are particularly motivated to investigate what may be the potential explanation for fatalities or injuries. In doing so, we deliver executable solutions for the policy makers at DOT to make better judgments for traffice in New York City.

In the context of big data such as VisionZedro project, good prediction is essential. Conventional approaches to prediction problems rely on trial and error search to evaluate models and machine learning algorithms through cross-validation may face challenges when data sets do not have sufficient sample size. The above direction may guide analysis to less-predictive variable sets due to overfitting in training set. 

# Data

We are looking at a variety of treatments executed by DOT in the past since 2002. The treatments are exposed in tree-based structures. For example, in streetscape elements, we may have Slow Turn Wedge interacting with Slow Turn Box as well as Right Turn Signal. We have also have Bike Lane and Raised Crosswalk together imposing an impact to the ongoing traffic. 

We collect and clean up a list of these treatments and simply mark them treatment 1, 2, ..., and so on. Each treatment in binary form so they are coded as 1 if there exists one and 0 otherwise.

The most intuitive approach is to present a correlation matrix and we can, though vaguely, visualize some patterns among variables in the data set. Conventionally, scholars will use a trial-and-error to detect variables. Sometimes more sophisticated mehods (such as Principle Components) will be attempted.

<p align="center">
  <img width="1000" src="https://github.com/yiqiao-yin/Investigation-of-High-order-Interactions-in-VisionZero-Project/blob/master/figs/corrplot.PNG">
</p>

For example, streetscape elements are presented as the following. We have a list of tree-like treatments imposed on the streets of New York City.

<p align="center">
  <img width="1000" src="https://github.com/yiqiao-yin/Investigation-of-High-order-Interactions-in-VisionZero-Project/blob/master/figs/elements.PNG">
</p>

For another example, it is common for people to have accidents in parking lot as well. We can certainly look at the treamtments such as Parking Lanes Removed, Parking Markings, and Parking Stripe.

<p align="center">
  <img width="1000" src="https://github.com/yiqiao-yin/Investigation-of-High-order-Interactions-in-VisionZero-Project/blob/master/figs/vehicle-parking.PNG">
</p>

# Lab Procedure

The following section we search for an algorithm to further explore our target, Attrition, which is measured by 0 if the employee stays and 1 if the employee leaves. The task is to deliver a solution, a trainable machine, such that we can predict a new candidate's probability of Attrition with high accuracy rate.

We test screened and engineered data set using common machine learning algorithms such as Bagging or Bootstrap Aggregation, Gradient Boosting Machine, Naive Bayes, Linear Model or Least Squares, Tree-based Algorithms (RF, iterative RF, Bayesian Additive Regression Tree or BART).

| Name | Test Result (Measured by Accuracy) |
| --- | --- |
| Bagging or Bootstrap Aggregation |	0.972 |
| Gradient Boosting Machine |	0.903 |
| Naive Bayes |	0.847 
| Linear Model or Least Squares |	0.898 |
| Random Forest |	0.903 |
| iterative Random Forest |	0.903 |		
| Bayesian Additive Regression Tree (BART) |	0.972 |	

Accuracy refers to the percentage of sum of true positive and true negative.

# Conclusion

After careful investigation of VisionZero Dataset, we proposed a generalized search algorithm to detect high-order interactions in this dataset as a screening technique to target correct model specification. This allows to achieve high performance for common machine learning algorithms.
