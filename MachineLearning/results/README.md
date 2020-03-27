# Machine Learning Results

### 1. Predicting gun acquisition
#### Performance Metrics
- *Precision*: Interpreted as "out of all the guns classified as stolen, what percentage was actually stolen?" High precision scores indicate a low rate of false positives meaning the model does well not misidentifying stolen guns.
- *Recall*: Understood as "out of all stolen guns, what percentage was correctly identified as stolen?" High recall scores indicate a high degree of sensitivity to identifying stolen guns meaning it does well selecting stolen guns from the mass.
- *Accuracy*: "Given all of the samples, what percentage was correctly classified?" It can be understood as the general effectiveness of the model.

#### Algorithm Performance
- *Logistic Regression*:
    + Average precision: 90%
    + Average recall: 76%
    + Accuracy score: 76.45%
- *Linear SVM*:
    + Average precision: 90%
    + Average recall: 78%
    + Accuracy score: 75.84%
- *Random Forest Classifier*:
    + Average precision: 92%
    + Average recall: 91%
    + Accuracy score: 77.55%
- *Deep Neural Network*
    + Average precision: 90%
    + Average recall: 91%
    + Accuracy score: 91.41%

#### Interpretations
When looking at this binary classification problem, and the original question being asked, the answer is yes, the dataset provides features that make it possible to predict the gun acquisition method with up to 91% confidence. In this situation, the Neural Network outperforms the other algorithms in pretty much every metric. It's worth noting that the DNN was trained and tested using the original imbalanced, unscaled dataset as compared to the other 3 models. This is becasue the dataset formats were not as compatible with the DNN as with the other models and the model produced an accuracy score of > 90% without the adjusted data. However, given the original dataset, the other algorithms did significantly worse overall and tended to be skewed towards the `stolen` class. 

### 2. Predicting suspect outcome
#### Performance Metrics
- *Precision*: "Out of all the suspects classified as a certain status, what percentage was correctly identified?" High precision scores means the model does well not misclassifying that specific status.
- *Recall*: "Out of all the suspects with a certain outcome, what percentage was correctly identified as that status?" High recall scores indicate the model does well recognizing that suspect outcome from the mass.
- *Accuracy*: "Given all of the samples, what percentage was correctly classified?" It can be understood as the general effectiveness of the model.

#### Algorithm Performance
- *Multinomial Logistic Regression*:
    + Accuracy score: 63.17%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.81		  |  0.46	  |
| 1			  |  0.25		  |  0.98	  |
| 2			  |  0.30		  |  0.45	  |

- *Linear SVM*:
    + Accuracy score: 63.06%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.81		  |  0.41	  |
| 1			  |  0.25		  |  1.00	  |
| 2			  |  0.29		  |  0.48	  |

- *Random Forest Classifier*:
    + Accuracy score: 56.42%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.79		  |  0.79	  |
| 1			  |  0.49		  |  0.54	  |
| 2			  |  0.38		  |  0.36	  |

- *Neural Network*:
    + Accuracy score: 72.58%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.74		  |  0.97	  |
| 1			  |  0.54		  |  0.45	  |
| 2			  |  1.00		  |  0.00	  |

#### Interpretations
Because this is a multi-class classification problem, it the overall accuracy score is perhaps the most indicative of model performance. This is because average recall and average precision begin to lose their significane when there are more than 2 classes. Therefore it's more important to look at the specific precision/recall scores of each class. What's interesting to note from these results is the similarity of the regression, SVM, and random forest methods compared to the neural network. Overall, no algorithm was very successful at predicting the outcome of the suspect (determined by an accuracy score > 75%). However, the neural network was the highest performing and upon further inspection it's clear the model was well trained at identifying if the suspect was class 0 (numerical value for arrested). This makes sense given that the original dataset was skewed with about 3x more samples of arrested suspects compared to the other two outcomes (1 symbolizing killed, 2 representing other). The stand out metric from the neural network is the 100% precision and 0% recall for "status-other". This seems to show the model never mispredicted a suspect who wasnt "status-other"; however, the 0% recall score indicates a calculation error and upon inspection of the confusion matrix, there are only 2 true positives, 0 false positives, and 452 false negatives. These values reveal that the algorithm is actually very good at classifying "status-other" suspects once identified, but terrible at recognizing them from the mass. Thus it's returned very few results but all of those results were labeled correctly

### 3. Predicting fatalities by suspect characteristics
#### Performance Metrics
- *Precision*: "Out of all the incidents classified with a certain number of fatalities, what percentage was correctly identified?" High precision scores means the model does well not misclassifying that specific fatality-outcome.
- *Recall*: "Out of all the incidents with a certain number killed, what percentage was correctly classified as having that many fatalities?" High recall scores indicate the model does well recognizing that "incident-fatility" class from the mass.
- *Accuracy*: "Given all of the samples, what percentage was correctly classified?" It can be understood as the general effectiveness of the model.

#### Algorithm Performance
- *Multinomial Logistic Regression*:
    + Accuracy score: 52.37%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.78		  |  0.95	  |
| 1			  |  0.26		  |  0.05	  |
| 2			  |  0.27		  |  0.57	  |

- *Linear SVM*:
    + Accuracy score: 52.33% 

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.78		  |  0.97	  |
| 1			  |  0.26		  |  0.04	  |
| 2			  |  0.27		  |  0.57	  |

- *Random Forest Classifier*:
    + Accuracy score: 54.98%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.82		  |  0.60	  |
| 1			  |  0.31		  |  0.43	  |
| 2			  |  0.09		  |  0.42	  |

- *Neural Network*:
    + Accuracy score: 77.36%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.78		  |  1.00	  |
| 1			  |  0.74		  |  0.19	  |
| 2			  |  0.79		  |  0.03	  |

#### Interpretations
The results again show the neural network outperforming the other three models this time by a significant degree with a difference of > 20% accuracy. It performed with very consistent precision accross the 3 classes meaning that it identified a high amount of releveant items for each class. However, the recall scores tell the bigger story at play. Firstly, the algorithm does very well (perfect in fact) at identifying relevant incidences with no fatalities. It's when looking at the models ability to discern 1 and 2 kill incidences that the model begins to fail. With a recall score of 3% for predicting 2-fatality incidences, it becomes clear that the model is skewed due to the imbalanced data and favors no-fatalitiy incidences given the larger proportion of that class in the training data. 

### 4. Predicting shooting severity by incident notes
#### Performance Metrics
*same as **Predicting fatalaties by suspect characteristics***

#### Algorithm Performance
Used the feature of incident *`notes`*
- *Naive Bayes*
- Predicting `n_killed`:
    + Accuracy score: 77.83%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.91		  |  0.83	  |
| 1			  |  0.53		  |  0.67	  |
| 2			  |  0.09		  |  0.15	  |

- Predicting `n_injured`:
    + Accuracy score: 70.83%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.84		  |  0.76	  |
| 1			  |  0.64		  |  0.72	  |
| 2			  |  0.14		  |  0.17	  |

Used the feature of *`incident_characteristics`* 
- *Naive Bayes*
- Predicting `n_killed`:
    + Accuracy score: 97.29%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  1.00		  |  0.99	  |
| 1			  |  0.88		  |  0.99	  |
| 2			  |  0.84		  |  0.02	  |

- Predicting `n_injured`:
    + Accuracy score: 44.09%

| Class		  |	Precision 	  |  Recall   |
|:-----------:|--------------:|----------:|
| 0			  |  0.54		  |  0.69	  |
| 1			  |  0.14		  |  0.09	  |
| 2			  |  0.00		  |  0.00	  |

#### Interpretations
This question produced a wide range of performances from the same model, a Naive Bayes predictor. Most notably, the model excelled when predicting the number of fatalities given the `incident_characteristics` with an accuracy score of > 97%. Upon inspecting the performance metrics, the model does tremendous at predicting incidences with 0 or 1 fatality. It's when predicting incidents with 2 fatalities that the performance dips considerably with a recall score of 2%. This indicates that the model is very poor at recognizing relevant incidents for the 2-fatality class. It's also interesting to note that given the same feature column, `incident_characteristics` the model does horribly, the worst performance of all, when predicting the number of injuries. Neither it's precision or accuracy scores are high enough to be considered significant (> 75%) for any class. 

