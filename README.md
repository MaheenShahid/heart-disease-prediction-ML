# ❤️ Heart Disease Prediction using Machine Learning

This project predicts heart disease presence using various supervised ML models on patient health data. Target:  
`0` = No Heart Disease, `1` = Heart Disease

---

## 📁 Dataset

 The dataset used is from Kaggle:  
 [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

- **Target Variable:** `target`
- **Features:** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

---

## 🧠 Models Used

| Model                | Type                 |
|----------------------|----------------------|
| Logistic Regression  | Linear Classifier    |
| KNN                  | Distance-Based       |
| Naive Bayes          | Probabilistic        |
| SVM                  | Margin-Based         |
| Decision Tree        | Tree-Based           |
| Random Forest        | Ensemble Trees       |
| Gradient Boosting    | Boosted Ensemble     |

---

## 📊 Evaluation

Models are evaluated on:
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1)

**Comparison Chart:**  
A seaborn barplot shows model-wise accuracy for easy visualization.

---

## ✅ Sample Results

| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression| 0.86     |
| KNN                | 0.81     |
| Naive Bayes        | 0.84     |
| SVM                | 0.83     |
| Decision Tree      | 0.77     |
| Random Forest      | 0.88     |
| Gradient Boosting  | 0.90     |

---

## 🛠️ Tools & Libraries

- Python
- scikit-learn
- pandas
- matplotlib
- seaborn
- VS Code

---

## 👩‍💻 Author

**Maheen Shahid**  
A machine learning enthusiast — exploring real-world healthcare applications!
