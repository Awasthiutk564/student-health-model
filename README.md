# Student Health Burnout Prediction

## Project Overview
This project focuses on predicting student burnout levels using machine learning.  
The goal is to classify students into **Low**, **Medium**, or **High** burnout categories based on academic, psychological, and lifestyle factors.

This project was developed as an **End Semester AIML Project**.

## Dataset
- **Dataset Name:** Student Mental Health and Burnout Dataset
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/sehaj1104/student-mental-health-and-burnout-dataset)
- **Total Records:** 150,000
- **Total Features:** 20
- **Target Variable:** `burnout_level`

## Problem Statement
Student mental health is affected by many factors such as stress, anxiety, sleep quality, academic pressure, and social support.  
The objective of this project is to build a machine learning model that predicts burnout level from the available student-related features.

## Machine Learning Task
- **Type:** Supervised Learning
- **Problem Type:** Multi-class Classification
- **Classes:** Low, Medium, High

## Features Used
Some important input features from the dataset are:
- age
- gender
- course
- year
- daily_study_hours
- daily_sleep_hours
- screen_time_hours
- stress_level
- anxiety_score
- depression_score
- academic_pressure_score
- financial_stress_score
- social_support_score
- physical_activity_hours
- sleep_quality
- attendance_percentage
- cgpa
- internet_quality

## Models Used
Two machine learning models were trained and evaluated:
- Logistic Regression
- Random Forest Classifier

## Project Workflow
1. Load the dataset
2. Perform basic data analysis
3. Check missing values
4. Visualize important patterns
5. Preprocess categorical and numerical features
6. Split data into training and testing sets
7. Train classification models
8. Evaluate model performance
9. Generate confusion matrix and plots

## Visualizations Generated
The project includes the following output graphs:
- Burnout Level Distribution
- Stress Level vs Burnout Level
- Anxiety Score by Burnout Level
- Random Forest Confusion Matrix
- Model Accuracy Comparison
- Feature Importance Graph

## Results
### Model Accuracy
- **Logistic Regression:** 32.77%
- **Random Forest:** 33.53%

### Observation
Random Forest performed slightly better than Logistic Regression.  
However, the overall accuracy remained low, which suggests that predicting burnout level is a challenging multi-class classification problem.

## Conclusion
This project shows how machine learning can be applied to student mental health analysis.  
Although the models did not achieve high accuracy, the project helped in understanding:
- data preprocessing
- classification techniques
- model evaluation
- visualization of results

## Future Scope
In future, the project can be improved by:
- applying advanced models like XGBoost or Gradient Boosting
- performing better feature engineering
- tuning hyperparameters
- testing deep learning approaches
- using more informative real-world features

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure
```text
student_health_model/
│── main.py
│── .gitignore
│── outputs/
│   ├── anxiety_vs_burnout.png
│   ├── burnout_distribution.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_rf.png
│   ├── feature_importance_rf.png
│   ├── model_accuracy_comparison.png
│   └── stress_vs_burnout.png

How to Run
Create and activate a virtual environment
Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
Place the dataset file as:
student_burnout.csv
Run:
python main.py

Now it is done 