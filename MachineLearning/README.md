## Machine Learning Overview

### Framing Questions
1. Can the provided metrics create a means of predicting whether each gun was stolen or not?
2. How about predicting the outcome of each suspect?
3. Using characteristics of the suspect, can the number of fatalities be predicted?
4. Is there a way to use the incident characteristics notes to predict the severity of the shooting?


### Summary

#### Universal Data Transformations
1. Read data in from PostgreSQL database hosted on AWS
2. Started with transforming the guns DataFrame   
    a. Binned the gun types into broad catagories; handgun, rifle, assault rifle, and shotgun   
    b. Dummy encoded the gun stolen and gun type columns 
3. Next, transformed the suspects DataFrame   
    a. Consolidated the status column into 6 outcomes   
    b. Dummy encoded the suspects gender, age group, and status
4. Finally, uploaded the transformed tables back into SQL database to be accessed in separate Jupyter Notebooks by the models

*see `Data_Transformations.ipynb` for code*

#### Question-Specific Process Description
1. **Predicting gun acquisition**:
    - Utilized supervised learning methods because dataset contains labels
    - Binary classification problem; either stolen or not
    - Started by importing guns and incidents table from AWS
    - Merged dataframes and selected for rows with target information (stolen/not-stolen)
    - Isolated features to be columns with geographic, politcal, and incident-specific information. This included: `state`, `congressional_district`. `state_house_district`, `state_senate_district`, `n_guns_involved`, `n_killed`, and `n_injured` 
    - Using the above features, explored a **Logistic Regression** algorithm:
        + Average precision: 90%
        + Average recall: 76%
        + Accuracy score: 76.45%
    - Implemented a **Linear SVM** algorithm:
        + Average precision: 90%
        + Average recall: 78%
        + Accuracy score: 75.84%
    - Explored a **Random Forest Classifier**:
        + Average precision: 92%
        + Average recall: 91%
        + Accuracy score: 77.55%
    - Explored a **Deep Neural Network** (10 input nodes, 20 hidden nodes in layer 1, 6 hidden nodes in layer 2, 1 output node)
        + Average precision: 91%
        + Average recall: 81%
        + Accuracy score: 81.29%

2. **Predicting suspect outcome**:
    - Have labeled dataset, can implement supervised learning algorithms
    - Have multiple outcomes so this becomes a multi-class problem 
    - Imported both suspects and incidents tables from AWS and merged the two dataframes
    - Binned status outcomes into 4 main categories: arrested, killed, other, and unknown
    - Dropped all status unknown entries to leave only rows with target data
    - Used the following features: `age`, `gender`, `state`, `congressional_district`. `state_house_district`, `state_senate_district`, `n_killed`, and `n_injured`
    - Started with a **Multinomial Logistic Regression** algorithm:
        + Average precision: 66%
        + Average recall: 49%
        + Accuracy score: 63.17%
    - Explored a **Linear SVM** algorithm:
        + Average precision: 66%
        + Average recall: 47%
        + Accuracy score: 63.06% 
    - Explored a **Random Forest Classifier**:
        + Average precision: 68%
        + Average recall: 68%
        + Accuracy score: 56.42%
    - Investigated a **Neural Network** (12 input nodes, 24 hidden nodes, 3 output nodes):
        + Average precision: 78%
        + Average recall: 73%
        + Accuracy score: 72.58%

3. **Predicting fatalities by suspect characteristics**:
    - Have labeled dataset so therefore utilized supervised learning algorithms
    - Imported suspects and incidents tables from AWS
    - Merged dataframes
    - Removed children from new dataframe (contained 0.35% of dataset and were recognized as outliers) 
    - Indentified features to be: `age`, `gender`, `state`, and `status`
    - Removed outlier targets which were incidents with `n_killed` > 2
    - Started with a **Multinomial Logistic Regression** algorithm:
        + Average precision: 64%
        + Average recall: 72%
        + Accuracy score: 52.37%
    - Explored a **Linear SVM** algorithm:
        + Average precision: 64%
        + Average recall: 73%
        + Accuracy score: 52.33% 
    - Explored a **Random Forest Classifier**:
        + Average precision: 68%
        + Average recall: 55%
        + Accuracy score: 54.98%
    - Investigated a **Neural Network** (6 input nodes, 12 hidden nodes, 3 output nodes):
        + Average precision: 77%
        + Average recall: 77%
        + Accuracy score: 77.36%

4. **Predicting shooting severity by incident notes**:
    - Used the feature of incident **`notes`** to predict `n_killed` and `n_injured`
        - Imported incidents table from AWS
        - Extracted `notes` as the feature and `n_killed`/`n_injured` as the targets
        - Created an NLP pipeline to determine "significance" of each word
        - Filtered outliers which were rows with `n_killed`/`n_injured` > 4 
        - Implemented a **Naive Bayes** model to predict the target given the feature vectors produced by the NLP pipeline
        - Predicting `n_killed`:
            + Average precision: 81%
            + Average recall: 78%
            + Accuracy score: 77.83%
        - Predicting `n_injured`:
            + Average precision: 72%
            + Average recall: 71%
            + Accuracy score: 70.83%
    - Used the feature of **`incident_characteristics`** to predict `n_killed` and `n_injured`
        - Imported incidents table from AWS
        - Extracted `incident_characteristics` as the feature and `n_killed`/`n_injured` as the targets
        - Created an NLP pipeline to determine "significance" of each word
        - Filtered outliers which were rows with `n_killed`/`n_injured` > 4 
        - Implemented a **Naive Bayes** model to predict the target given the feature vectors produced by the NLP pipeline
        - Predicting `n_killed`:
            + Average precision: 97%
            + Average recall: 97%
            + Accuracy score: 97.29%
        - Predicting `n_injured`:
            + Average precision: 36%
            + Average recall: 44%
            + Accuracy score: 44.09%
