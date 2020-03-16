## ETL Process

1. Started by importing dataset into Pandas DataFrame
2. Proceeded to filter dataset for only relevant columns
3. Converted columns to proper datatypes
4. Created a DataFrame for all incidents containing the date, location, and incident characteristics
5. Created a seperate suspects DataFrame containing relevant columns for each participant   
    a. Exploded each column containing more than one datapoint per row   
    b. Merged exploded columns into single dataframe
6. Again, created another DataFrame for gun information   
    a. Exploded each column and merged them
7. Exported a sample of each DataFrame to a `.csv` file located in `/sample_transformations`
8. Finally. uploaded each transformed DataFrame into a PostgreSQL database hosted by AWS
