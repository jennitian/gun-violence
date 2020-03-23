## Machine Learning Tracking

### Framing Questions
1. Can the provided metrics create a means of predicting whether each gun was stolen or not?
2. How about predicting the outcome of each suspect?
3. Using characteristics of the suspect, can the number of fatalities be predicted?
4. Is there a way to use the incident characteristics notes to predict the severity of the shooting?

### Data Transformation
1. Read data in from PostgreSQL database hosted on AWS
2. Started with transforming the guns DataFrame   
    a. Binned the gun types into broad catagories ie. handgun, rifle, etc.   
    b. Dummy encoded the gun stolen column 
3. Next, transformed the suspects DataFrame   
    a. Consolidated the status column into 6 outcomes   
    b. Dummy encoded the suspects gender, age group, and status
4. Finally, uploaded the transformed tables back into SQL database to be accessed in separate Jupyter Notebooks

### Algorithms
(:moneybag: *indicates best performing model*)
1. Predicting gun stolen/not-stolen   
    a. **Logistic Regression**   
        - Created classifer model (`max_iter` = 200)   
        - Fit model with randomly oversampled & normalized training data   
        - Produced a balanced accuracy score of 76.719%   
    b. **LinearSVC**   
        - Created model   
        - Fit model with randomly oversampled & normalized training data    
        - Produced balanced accuracy score of 75.844%   
    c. **Random Forest Classifier**   
        - Created model (`n_estimators` = 200)   
        - Fit model with balanced training data   
        - Produced balanced accuracy score of 77.644%   
    d. **Neural Network** :moneybag:   
        - Defined model: 10 hidden nodes in first layer with `'relu'` activation, 6 hidden nodes in second layer with `'relu'` activation, 1 output node with `'sigmoid'` activation   
        - Compiled model: `'binary_crossentropy'` loss function   
        - Trained model (100 epochs)   
        - Balanced accuracy score of 80.343%   

2. Predicting suspect outcome   
    a. **Multinomial Logistic Regression**   
        - Created model (`max_iter` = 200)   
        - Fit model with randomly oversampled & normalized training data   
        - Made predictions using test data   
        - Produced balanced accuracy score of 62.988%   
    b. **LinearSVC**   
        - Created model   
        - Fit model with randomly oversampled & normalized training data   
        - Made predictions on scaled test data   
        - Produced accuracy score of 62.350%   
    c. **Random Forest Classifier**   
        - Created model (`n_estimators` = 200)   
        - Fit model on training data   
        - Made predictions using test data   
        - Produced balanced accuracy score of 67.175%   
    d. **Neural Network** :moneybag:   
        - Defined model: 7 input nodes, 14 hidden nodes on first layer with `'relu'` activation, 3 output nodes with `'softmax'` activation   
        - Compiled model: `'sparse_categorical_crossentropy'` loss function   
        - Fit model using training data (100 epochs)   
        - Produced balanced accuracy score of 72.827%   

3. Predicting number of fatalities
    a. **Multinomial Logistic Regression**   
        - Created model (`max_iter` = 200)   
        - Fit model with randomly oversampled & normalized training data   
        - Made predictions using test data   
        - Produced balanced accuracy score of 52.373%   
    b. **LinearSVC**   
        - Created model   
        - Fit model with randomly oversampled & normalized training data   
        - Made predictions on scaled test data   
        - Produced accuracy score of 52.328%   
    c. **Random Forest Classifier**   
        - Created model (`n_estimators` = 200)   
        - Fit model on training data   
        - Made predictions using test data   
        - Produced balanced accuracy score of 54.983%   
    d. **Neural Network** :moneybag:   
        - Defined model: 6 input nodes, 12 hidden nodes on first layer with `'relu'` activation, 3 output nodes with `'softmax'` activation   
        - Compiled model: `'sparse_categorical_crossentropy'` loss function   
        - Fit model using training data (50 epochs)   
        - Produced balanced accuracy score of 77.420%   

4. NLP Prediction   
	a. **Analyzing Notes**   
	    i. Created feature vector of notes column   
	    ii. Created NLP pipeline   
	    iii. Used a Naive Bayes algorithm to predict label from feature vector   
	    *Predicting n_killed*      
	    - Produced accuracy score of 78.862%   
	    *Predicting n_injured*   
	    - Produced accuracy score of 70.845%   
	b. **Analyzing Incident Characteristics**   
		i. Created feature vector of incident_characteristics column   
	    ii. Created NLP pipeline   
	    iii. Used a Naive Bayes algorithm to predict label from feature vector   
	    *Predicting n_killed*      
	    - Produced accuracy score of 96.552%   
	    *Predicting n_injured*   
	    - Produced accuracy score of 38.985%   
