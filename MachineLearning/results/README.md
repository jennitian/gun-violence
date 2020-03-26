## Machine Learning Results

### Question-Specific Model Breakdown

1. **Predicting gun acquisition**:

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
        + Average precision: 91%
        + Average recall: 81%
        + Accuracy score: 81.29%

2. **Predicting suspect outcome**:
    - *Multinomial Logistic Regression*:
        + Average precision: 66%
        + Average recall: 49%
        + Accuracy score: 63.17%
    - *Linear SVM*:
        + Average precision: 66%
        + Average recall: 47%
        + Accuracy score: 63.06% 
    - *Random Forest Classifier*:
        + Average precision: 68%
        + Average recall: 68%
        + Accuracy score: 56.42%
    - *Neural Network*:
        + Average precision: 78%
        + Average recall: 73%
        + Accuracy score: 72.58%

3. **Predicting fatalities by suspect characteristics**:
    - *Multinomial Logistic Regression*:
        + Average precision: 64%
        + Average recall: 72%
        + Accuracy score: 52.37%
    - *Linear SVM*:
        + Average precision: 64%
        + Average recall: 73%
        + Accuracy score: 52.33% 
    - *Random Forest Classifier*:
        + Average precision: 68%
        + Average recall: 55%
        + Accuracy score: 54.98%
    - *Neural Network*:
        + Average precision: 77%
        + Average recall: 77%
        + Accuracy score: 77.36%

4. **Predicting shooting severity by incident notes**:
    - Used the feature of incident *`notes`*
        - *Naive Bayes*
        - Predicting `n_killed`:
            + Average precision: 81%
            + Average recall: 78%
            + Accuracy score: 77.83%
        - Predicting `n_injured`:
            + Average precision: 72%
            + Average recall: 71%
            + Accuracy score: 70.83%
    - Used the feature of *`incident_characteristics`* 
        - *Naive Bayes*
        - Predicting `n_killed`:
            + Average precision: 97%
            + Average recall: 97%
            + Accuracy score: 97.29%
        - Predicting `n_injured`:
            + Average precision: 36%
            + Average recall: 44%
            + Accuracy score: 44.09%
