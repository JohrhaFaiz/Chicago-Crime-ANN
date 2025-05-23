
# Future Crimes Prediction Using Deep Artificial Neural Networks (ANN)

This project combines research-backed methodology and practical implementation to predict crime categories using historical crime data from the city of Chicago. It implements two supervised Artificial Neural Network (ANN) algorithms: Backpropagation (BP) and QuickProp, aligned with a working model built using the fastai library.

---

##  Objective

To develop a predictive model for classifying crimes based on historical data using deep learning ANN architectures. This effort aims to assist law enforcement in anticipating potential crime types based on available attributes.

---

##  Dataset Overview

- Source: Chicago Police Department CLEAR System  
- Years Covered: 20012020  
- Size: 1.6 GB  
- Records: 7.2 million  
- Format: CSV


> **Note:**  
> The raw dataset `rows.csv` is not stored locally in this repository.  
> It is automatically downloaded from the official Chicago data portal using the following command inside the notebook:
!wget "https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv"



### Original → Reduced Crime Categories

| Group                   | Merged Types Example                             |
|-------------------------|--------------------------------------------------|
| Forbidden Practices     | Narcotics, Prostitution, Gambling                |
| Theft                   | Robbery, Burglary, Deceptive Practice           |
| Criminal Assault        | Homicide, Sexual Offense, Child Offense         |
| Public Peace Violation  | Arson, Intimidation, Weapons                    |


##  Data Preprocessing

- Dropped unnecessary columns: ID, Case Number, FBI Code, etc.
- Selected categorical features: Block, Primary Type, Location Description, Beat, District, Ward, Community Area, Year
- Filtered dataset to entries with Year  2020
- Imputed missing values:
  - Filled Location Description with OTHER
  - Filled District using the most common value
  - Imputed Ward and Community Area based on common Block

---

##  Model Design

- Implemented using fastais TabularModel
- Categorical variables embedded into dense vectors
- Two hidden layers: [128, 64]
- Output: Softmax classifier with 4 classes

### Training Configuration

### Training Configuration

| Parameter     | Value                    |
|---------------|---------------------------|
| Epochs        | 1                         |
| Batch Size    | 64                        |
| Optimizer     | Adam / QuickProp          |
| Split         | 95% training / 5% validation |

---

## Performance Metrics

| Model              | Accuracy (%) | Precision | Recall | F1 Score |
|--------------------|--------------|-----------|--------|----------|
| BP (Backpropagation) | 46.47        | 18.83     | 11.22  | 10.28    |
| QuickProp          | 46.38        | 18.24     | 11.25  | 10.29    |

> **Note:** Results are affected by data imbalance and limited features.


---

##  Key Challenges

- Data Imbalance: Theft crimes dominate the dataset (38)
- Limited Features: Lack of behavioral or contextual variables such as time, weather, or social patterns
- Model Limitation: Accuracy plateaus due to minimal discriminative features

---

##  Future Improvements

- Add continuous features (e.g., time of day, weather, police response time)
- Use RNNs or CLSTM for spatiotemporal patterns
- Implement sampling techniques to balance class distribution
- Deploy as an API or interactive dashboard for crime analysts

---

## Chicago Crime Data Visualization (EDA Notebook)

The notebook 'chicago_crime_charts.ipynb' provides comprehensive **exploratory data analysis (EDA)** for the Chicago crime dataset , focusing on visual trends and insights before modeling.

### What It Does:
- Loads large-scale parquet crime data using **Dask** for memory-efficient processing.
- Displays data stats and structural overview.
- Uses **Matplotlib** and **Seaborn** to generate clean, readable charts.
- Visualizes:
  - Crime frequency over time
  - Most common crime types
  - Hotspot locations
  - Time-based distributions (by hour, day, etc.)

### Why It Matters:
This notebook supports the deep learning model by helping:
- Understand data distributions and patterns
- Identify influential features for classification
- Reveal imbalances or anomalies in the dataset

It serves as a powerful visual companion to the ANN training process.
"""
