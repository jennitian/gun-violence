# Machine Learning Models

(:moneybag: *indicates best performing model*)
## Predicting gun stolen/not-stolen
### Data Preprocessing
- Connected to AWS PostreSQL database using `sqlalchemy` within a Jupyter Notebook
- Imported the ML-transformed dataset of guns used in the incidents as well as the dataset of all incidents
- Merged the two datasets into a single DataFrame and filtered the rows to only include ones with entries for gun acquisition (stolen/not-stolen)
- Dropped all columns except for desired features: `state`, `congressional_district`. `state_house_district`, `state_senate_district`, `n_guns_involved`, `n_killed`, and `n_injured`
- Encoded the states to have numerical representations in order to be processed by the ML models
- Examined the IQR (Inter-Quartile Range) of each column to determine the spread of the data
- Dropped the `state_house_district` feature because it had too large of a spread making it difficult to discern outliers
- Separated the features and target (`stolen`) into seperate data structures 
- Split into training and testing datasets using a random, stratified method that created test and train datasets for the features and target. The split was 75% train and 25% test and it was stratified meaning the proportion of values is the same in the test and train sets
- Randomly oversampled the training dataset to account for the imbalance in the data. The total number of training data points was 21,160
- Finally, scaled the feature data to account for large spread

### Algorithms
1. *Logistic Regression*
	- Created classifer model 
		+ `max_iter` of 200
		+ 'lbfgs' `solver`
	- Fit model with randomly oversampled & normalized training data
	- Produced balanced accuracy score of 76.719%   
2. *LinearSVC*
    - Created SVC model with `linear` kernel  
    - Fit model with randomly oversampled & normalized training data    
    - Produced balanced accuracy score of 75.844%   
3. *Random Forest Classifier*
    - Created model (`n_estimators` = 200)   
    - Fit model with balanced and scaled training data   
    - Produced balanced accuracy score of 77.644%   
4. *Deep Neural Network* :moneybag:   
    - Defined model: 
    	+ 20 hidden nodes in first layer with `'relu'` activation
    	+ 5 hidden nodes in second layer with `'relu'` activation
    	+ 1 output node with `'sigmoid'` activation
    - Hidden layers were determined based upon multiples of the number of input features (10)
    - Output activation selected because of the binary classification nature of the question posed
    - Compiled model: `'binary_crossentropy'` loss function   
    - Trained model (100 epochs)   
    - Balanced accuracy score of 80.343%   

## Predicting suspect outcome
### Data Preprocessing
- Connected to AWS PostreSQL database using `sqlalchemy`
- Imported the ML-transformed dataset of suspects involved in the incidents as well as the dataset of all incidents
- Merged the two datasets into a single DataFrame
- Binned the suspect statuses' into 4 groups; arrested, killed, other, and unknown
- Encoded the state column to have a numerical representation of each state
- Dropped unnecessary columns to leave only desired features of `age`, `gender`, `state`, `congressional_district`. `state_house_district`, `state_senate_district`, `n_killed`, and `n_injured`
- Dropped all rows with status unknown label
- Calculated the IQR of each column to examine spread of data
- Dropped all child suspects (defined as being between the ages of 0-11 which constituted 0.35% of dataset and were recognized as outliers based on the 1.5 * IQR rule of thumb)
- Dropped high end of ages identified as outliers (defined as being older than 100 years old)
- Separated features and target (`status`)
- Implemented a random train/test split which returned 75% of the original data as a training set and 25% as a test set
- Performed random oversampling to account for the imbalance of targets
- The result was 143,115 training data points split evenly between the 3 target classes of arrested, killed, and other
- Finally, implemented a standard scaler to account for the larger range of values present in the district columns compared to other features

### Algorithms
1. *Multinomial Logistic Regression*
    - Created model (`multi_class` = 'multinomial', solver` = 'newton-cg', `max_iter` = 200)   
    - Defined the model as multinominal because this is a multi-class problem with there being 3 outcomes
    - Fit model with randomly oversampled & normalized training data
    - Made predictions using scaled test data
    - Produced balanced accuracy score of 62.988%
2. *LinearSVC*
    - Created LinearSVC model
    - Fit model with randomly oversampled & normalized training data
    - Made predictions on scaled test data
    - Produced accuracy score of 62.350%
3. *Random Forest Classifier*
    - Created model (`n_estimators` = 200)
    - Fit model on training data
    - Made predictions using test data
    - Produced balanced accuracy score of 67.175%
4. *Neural Network* :moneybag:
    - Defined model:
    	+ 14 hidden nodes on first layer with `'relu'` activation
    	+ 3 output nodes with `'softmax'` activation
    - Hidden layers were determined based upon multiples of the number of input features (7)
    - Output activation selected because the fact there are 3 target classes in this problem
    - Compiled model: `'sparse_categorical_crossentropy'` loss function
    - Fit model using training data (100 epochs)
    - Produced balanced accuracy score of 72.827%

## Predicting number of fatalities
#### Data Preprocessing
- Connected to AWS PostreSQL database using `sqlalchemy`
- Imported the ML-transformed dataset of suspects involved in the incidents along with the dataset of all incidents
- Merged the two datasets into a single DataFrame
- Grouped the suspect statuses' into 4 categories; arrested, killed, other, and unknown
- Encoded the state column to have a numerical representation of each state
- Dropped unnecessary columns to leave only desired features of `age`, `gender`, `state`, and `status`
- Dropped all rows with status unknown label as well as rows containing any null values
- Calculated the IQR of each column to examine spread of data
- Dropped all child suspects (defined as being between the ages of 0-11 which constituted 0.35% of dataset and were recognized as outliers based on the 1.5 * IQR rule of thumb)
- Dropped high end of ages as well (defined as being older than 100 years old)
- Inspected target column and found that 99.36% of incidents resulted in < 3 fatalities, filtered data to reflect this cutoff
- Separated features and target (`n_killed`)
- Implemented a random train/test split with 75% of the original data as a training set and 25% as a test set
- Performed random oversampling to account for the imbalance of targets
- The result was 177,288 training data points split evenly between the 3 target classes of a 0 fatalities, 1 fatality, and 2 fatalities
- Finally, implemented a standard scaler to account for the larger range of values present in the age and state columns compared to other features

### Algorithms:
1. *Multinomial Logistic Regression*
    - Created model (`multi_class` = 'multinomial', solver` = 'newton-cg', `max_iter` = 200) 
    - Defined the model as multinominal because this is a multi-class problem with there being 3 classes of "incident-fatalities": 0, 1, or 2  
    - Fit model with randomly oversampled & normalized training data   
    - Made predictions using scaled test data   
    - Produced balanced accuracy score of 52.373%   
2. *LinearSVC*
    - Created LinearSVC model   
    - Fit model with randomly oversampled & normalized training data   
    - Made predictions on scaled test data   
    - Produced accuracy score of 52.328%   
3. *Random Forest Classifier*
    - Created model (`n_estimators` = 200)   
    - Fit model on balanced and scaled training data   
    - Made predictions using scaled test data   
    - Produced balanced accuracy score of 54.983%   
4. *Neural Network* :moneybag:
    - Defined model: 
    	+ 12 hidden nodes on first layer with `'relu'` activation
    	+ 3 output nodes with `'softmax'` activation   
    - Hidden layers were determined based upon multiples of the number of input features (6)
    - Output activation selected because there are 3 target classes in this problem
    - Compiled model: `'sparse_categorical_crossentropy'` loss function   
    - Fit model using training data (50 epochs)   
    - Produced balanced accuracy score of 77.420%   

## NLP Prediction
### Data Preprocessing
- Created a Spark session using `pyspark` 
- Connected to AWS PostgreSQL databasing using Spark
- Imported the incidents table into a Spark dataframe
- Dropped unnecessary columns to leave only desired features/targets: `notes`, `incident_characteristics`, `n_killed`, and `n_injured`
- Created length feature that totaled the words contained within `notes`/`incident_characteristics`
- Inspected spread of targets and determined that 99.36% of incidents resulted in < 3 fatalities, filtered data to reflect this cutoff
- Created feature vector of respective feature (`notes`/`incident_characteristics`) column   
- Created NLP pipeline: tokenized the feature → removed stop words → hashed the remaining values → claculated TFIDF (Term Frequency - Inverse Document Frequency)
- Separated dataset into training (70%) and test (30%) set using a random split
- Used a Naive Bayes algorithm to predict label from feature vector

### Algorithms
1. Analyzing `notes`
    - Predicting `n_killed` produced accuracy score of 78.862%   
    - Predicting `n_injured` produced accuracy score of 70.845%   
2. Analyzing `incident_characteristics`
    - Predicting `n_killed` produced accuracy score of 96.552%   
    - Predicting `n_injured` produced accuracy score of 38.985%   
