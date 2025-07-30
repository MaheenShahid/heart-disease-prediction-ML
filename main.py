import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


#heart disease prediction using supervised learning models

#Logistic Regression

df = pd.read_csv("D:\\Downloads\\heart-disease-prediction-ML\\heart_disease.csv")

#check if there are missing values
print(df.isnull().sum())

#displaying first five rows
print(df.head())

#dataset info
print(df.info())

X = df.drop("target", axis = 1)
Y = df["target"]

X_train, X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)

# ========== Logistic Regression ==========

model = LogisticRegression(max_iter=1000)

model.fit(X_train , Y_train)

log_pred = model.predict(X_test)

print("\n 🔹Logistic Regression")
# Accuracy 
print("Accuracy: ", accuracy_score(Y_test,log_pred))

# Confusion Matrix 
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, log_pred))

# Detailed performance report
print("\nClassification Report:\n", classification_report(Y_test, log_pred))

# ========== K-Nearest Neighbors ==========
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
knn_pred = knn_model.predict(X_test)
print("\n🔹 K-Nearest Neighbors")
print("Accuracy:", accuracy_score(Y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, knn_pred))
print("Classification Report:\n", classification_report(Y_test, knn_pred))

# ========== Naive Bayes ==========
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)
nb_pred = nb_model.predict(X_test)
print("\n🔹 Naive Bayes")
print("Accuracy:", accuracy_score(Y_test, nb_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, nb_pred))
print("Classification Report:\n", classification_report(Y_test, nb_pred))

# ========== Support Vector Machine ==========
svm_model = SVC()
svm_model.fit(X_train, Y_train)
svm_pred = svm_model.predict(X_test)
print("\n🔹 Support Vector Machine (SVM)")
print("Accuracy:", accuracy_score(Y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, svm_pred))
print("Classification Report:\n", classification_report(Y_test, svm_pred))

# ========== Decision Tree ==========
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
dt_pred = dt_model.predict(X_test)
print("\n🔹 Decision Tree")
print("Accuracy:", accuracy_score(Y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, dt_pred))
print("Classification Report:\n", classification_report(Y_test, dt_pred))

# ========== Random Forest ==========
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)
print("\n🔹 Random Forest")
print("Accuracy:", accuracy_score(Y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, rf_pred))
print("Classification Report:\n", classification_report(Y_test, rf_pred))

# ========== Gradient Boosting ==========
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, Y_train)
gb_pred = gb_model.predict(X_test)

print("\n🔹 Gradient Boosting Classifier")
print("Accuracy:", accuracy_score(Y_test, gb_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, gb_pred))
print("Classification Report:\n", classification_report(Y_test, gb_pred))

# ========== Visualizations ==========
# Correlation Heatmap
# plt.figure(figsize=(12,8))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Feature Correlation Heatmap')
# plt.show()

# # Count Plot: Target variable
# sns.countplot(x='target', data=df)
# plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
# plt.show()

# ========== Model Comparison Chart ==========

# Store model names and their accuracy scores

model_names = [
    "Logistic Regression", 
    "KNN", 
    "Naive Bayes", 
    "SVM", 
    "Decision Tree", 
    "Random Forest", 
    "Gradient Boosting"
]

accuracy_scores = [
    accuracy_score(Y_test, log_pred),
    accuracy_score(Y_test, knn_pred),
    accuracy_score(Y_test, nb_pred),
    accuracy_score(Y_test, svm_pred),
    accuracy_score(Y_test, dt_pred),
    accuracy_score(Y_test, rf_pred),
    accuracy_score(Y_test, gb_pred)
]

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x=model_names, y=accuracy_scores, palette='viridis')
plt.ylabel("Accuracy Score")
plt.xlabel("Machine Learning Models")
plt.title("Model Comparison - Accuracy Scores")
plt.show()
