# Neural Network Challenge - Employee Attrition and Department Prediction

## Project Overview
This project focuses on developing a branched neural network to assist the HR department in two critical areas of workforce management:  
1. **Attrition Prediction** – Predicting the likelihood of employees leaving the company.  
2. **Department Fit Prediction** – Determining which department best suits each employee.  

Employee retention and internal mobility are key drivers of organizational stability and growth. High turnover rates can lead to significant costs in recruitment, onboarding, and training, while misalignment in employee roles can impact productivity and morale. This project aims to proactively address these challenges by building a model that can predict which employees are at risk of leaving and recommend alternative departments where their skills may be better utilized.  

By leveraging machine learning techniques, the model processes a wide array of employee data—such as performance reviews, tenure, demographics, and job roles—to make predictions. The dual-output neural network provides insights into both attrition risk and departmental fit, empowering HR teams to make data-driven decisions to improve employee satisfaction and retention.  

---

## Coding Language and Environment
- **Language:** Python  
- **Frameworks/Libraries:** TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib
- **Development Environment:** Google Colab (recommended for GPU acceleration and ease of use)  
- **File Format:** Jupyter Notebook (`attrition.ipynb`)  

Python was chosen for this project due to its rich ecosystem of libraries and frameworks, enabling efficient development and deployment of machine learning models. TensorFlow and Keras serve as the foundation for constructing and training the neural network. Pandas and NumPy streamline data manipulation and preprocessing, while Scikit-learn handles encoding, scaling, and performance evaluation. Matplotlib is employed for visualizing key metrics.  

Google Colab was selected as the primary development environment to take advantage of its GPU support and Jupyter Notebook interface, ensuring faster training times and seamless collaboration.  

---

## Approach
The neural network is designed to handle two outputs simultaneously using a branched architecture. A shared input layer processes employee features, which then branches into two separate outputs:  
- **Attrition Prediction (Binary Classification):** Determines whether an employee is likely to leave.  
- **Department Fit (Multi-Class Classification):** Predicts the most suitable department for each employee.  

This structure allows the model to learn from overlapping data patterns, enhancing prediction accuracy across both tasks. The dataset consists of employee attributes such as job performance, job role, salary, department, and demographic information. The data is preprocessed, encoded, and scaled before being split into training and testing sets.  

---

## Project Breakdown
The project consists of three major components:

### 1. Preprocessing the Data
Data preprocessing is essential for achieving reliable and accurate model predictions. This phase includes:  
- **Importing and Inspecting Data:** Viewing the first few rows and analyzing key statistics.  
- **Feature Selection:** Choosing 10 relevant columns from the dataset for input features.  
- **Encoding and Scaling:** Converting categorical data to numeric using OneHotEncoder and scaling using StandardScaler.  
- **Train-Test Split:** Dividing the data into training and testing sets to evaluate performance on unseen data.  

### 2. Model Creation and Training
A branched neural network is constructed to handle the dual output tasks. The architecture consists of:  
- **Shared Hidden Layers:** Two dense layers that extract shared patterns between attrition and department prediction.  
- **Output Branches:**  
  - **Attrition Branch:** A sigmoid-activated output node for binary classification.  
  - **Department Branch:** A softmax-activated output node for multi-class classification.  
- **Compilation and Training:** The model is compiled using separate loss functions for each branch (binary cross-entropy for attrition, categorical cross-entropy for department prediction) and trained over multiple epochs.  

### 3. Model Evaluation and Improvement
Model performance is evaluated using test data to assess accuracy and loss for both outputs.  
- **Attrition Accuracy** – Measures the correctness of binary attrition predictions.  
- **Department Accuracy** – Assesses how well the model classifies employees into the correct departments.  

---

## Initial Model Design
The initial model utilized **sigmoid activation** for both attrition and department outputs. This resulted in the following performance:  
- **Attrition Prediction Accuracy:** 84.2%  
- **Department Prediction Accuracy:** 50%  
- **Overall Model Loss:** 4.18  
- **Attrition Loss:** 0.86  
- **Department Loss:** 3.32  

While attrition prediction accuracy was high, the department accuracy lagged significantly, indicating that sigmoid was not well-suited for multi-class classification.  

---

## Updated Approach
To improve department prediction, the activation function for the department output was changed to **softmax**. This adjustment resulted in the following performance:  
- **Attrition Prediction Accuracy:** 80.7%  
- **Department Prediction Accuracy:** 52.99%  
- **Overall Model Loss:** 3.51  
- **Attrition Loss:** 0.93  
- **Department Loss:** 2.58  

---

## Key Results
| Metric                        | Initial Results (Sigmoid) | Updated Results (Softmax) |  
|------------------------------|--------------------------|---------------------------|  
| Attrition Prediction Accuracy | 84.2%                    | 80.7%                     |  
| Department Prediction Accuracy| 50%                      | 52.99%                    |  
| Overall Model Loss            | 4.18                     | 3.51                      |  
| Attrition Loss                | 0.86                     | 0.93                      |  
| Department Loss               | 3.32                     | 2.58                      |  

---

## Activation Functions Used and Rationale  
Initially, **sigmoid activation** was used for both outputs. Sigmoid is effective for binary classification but limited for multi-class tasks. Switching to **softmax** for the department output improved performance by producing a probability distribution across classes. This allowed the model to select the class with the highest probability confidently.  

---

## Model Improvement Recommendations  
- **Feature Engineering:** Introduce more relevant employee metrics (skills, projects, etc.) to improve department fit predictions.  
- **Class Imbalance:** Apply SMOTE or oversampling techniques to address imbalanced department data.  
- **Regularization:** Use dropout layers and L2 regularization to reduce overfitting.  
- **Embedding Layers:** Represent categorical data more effectively using embeddings.  
- **Hyperparameter Tuning:** Optimize the number of neurons, layers, and learning rate to enhance accuracy further.  

---

## Summary
This project demonstrates the potential of neural networks to address HR challenges like attrition and employee alignment. By refining the activation functions and model architecture, significant improvements in prediction accuracy were achieved. Further enhancements through data augmentation and hyperparameter tuning are expected to boost performance even more.

 
## Resources 
I attended a tutoring session to review the initial code I had written and I subsequently revised a few sections of the code for better model performance.
