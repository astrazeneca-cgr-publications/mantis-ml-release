### Data compilation
Build feature table dynamically based on disease-relevant query terms provided by the user/researcher.


### Automaticlaly pre-processes feature table
Specifically:
- Drop duplicate and/or near-zero-variance features 
- One-hot encode categorical variables 
- Feature standardisation 
- Filter out highly-correlated features (based on user-defined threshold; default: 0.80) 
- Generate plots for Exploratory Analysis, such as histograms with distribution of numerical and/or categorical features in the two classes (positive/unlabelled).
