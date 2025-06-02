# 📊 Customer Satisfaction Classification using Gradient Boosting

This project analyzes customer satisfaction based on structured feedback collected using a **Likert scale**. The goal is to automatically classify whether a customer is **satisfied or not**, using **supervised machine learning** models trained on key service quality factors.

---

## 📁 Dataset Overview

- **File**: `Dados-SLA.csv`
- **Format**: CSV without headers
- **Records**: 119 clients
- **Columns**:  
  - 1 client identifier  
  - 11 service evaluation factors  
  - 1 overall satisfaction score (target variable)

Each item is rated using a **Likert scale from 1 (strongly disagree) to 5 (strongly agree)**.

---

## 🎯 Project Objective

The objective is to build a **predictive model** capable of identifying dissatisfied customers based on critical quality-of-service attributes. The insights generated can assist decision-makers in:

- Enhancing **service level agreements (SLA)**
- Prioritizing **areas for improvement**
- Minimizing **customer churn**
- Applying **operational research principles** for continuous improvement

---

## 🧠 Concepts from Operational Research

This project touches several important concepts from **Operational Research (OR)**:

| Concept | Application |
|--------|-------------|
| **Decision Support** | Predicting client satisfaction to support managers |
| **Multicriteria Analysis** | Evaluating multiple service dimensions simultaneously |
| **Optimization** | Improving resource allocation based on weak points |
| **Data-Driven Modeling** | Using real data for supervised learning and pattern detection |

---

## 📊 Variables and Likert Scale

All features and the target satisfaction score are measured using a **Likert-type scale (1–5)**. This scale captures subjective judgments such as satisfaction, quality, or reliability.

The classification target is created as follows:

- `Satisfied` = 1 if the overall score is **4 or 5**
- `Not Satisfied` = 0 if the score is **3 or below**

---

## 🔬 Machine Learning Approach

The following machine learning steps are applied:

### 1. **Preprocessing**
- Data reshaping
- Column renaming
- Feature scaling using `MinMaxScaler`

### 2. **Model Training**
- Using **Gradient Boosting Classifier** with 300 estimators
- Training on **all features**

### 3. **Evaluation**
- Accuracy score
- Classification report
- Confusion matrix (raw and normalized)
- Display of **misclassified clients**

### 4. **Feature Importance**
- Analyzing which features most impact the satisfaction prediction
- Visualizing sorted importance table

---

## 🧪 Technologies Used

- **Python 3.10+**
- `pandas`, `numpy` – data handling
- `seaborn`, `matplotlib` – data visualization
- `scikit-learn` – machine learning

---

## 🚀 How to Run

1. Place `Dados-SLA.csv` in the same directory as the script or notebook.
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
3. Run the script:
   ```bash
   python analyze_satisfaction.py

---

## 📌 Applications

This methodology is applicable to a wide range of real-world scenarios:
- Evaluating B2B or B2C satisfaction
- Auditing SLA compliance
- Building intelligent feedback dashboards
- Improving service delivery in logistics, IT, or support operations

---

## 🧾 License

This project is educational and open for academic and research purposes. Please cite or refer to the author if reused.

---

✍️ Author

Developed by Gabriel Ribeiro as part of research in Operational Research, Machine Learning, and Customer Satisfaction Modeling.