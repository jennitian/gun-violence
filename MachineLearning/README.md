## Machine Learning Tracking

### Framing Questions
1. Can the provided metrics provide a mean of predicting whether each gun was stolen or not?
2. How about predicting the outcome of each suspect?
3. Is there a way to use the incident characteristics notes to predict the severity of the shooting?

### Data Transformation
1. Read data in from PostgreSQL database hosted on AWS
2. Started with transforming the guns DataFrame   
    a. Binned the gun types into broad catagories ie. handgun, rifle, etc.   
    b. Dummy encoded the gun stolen column 
3. Next, transformed the suspects DataFrame   
    a. Consolidated the status column into 6 outcomes   
    b. Dummy encoded the suspects gender, age group, and status
4. Finally, uploaded the transformed tables back into SQL database