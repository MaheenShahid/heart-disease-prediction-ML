import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


#heart disease prediction 

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

model = LogisticRegression(max_iter=1000)

model.fit(X_train , Y_train)

predict_heart_disease = model.predict(X_test)

# Accuracy 
print("Accuracy: ", accuracy_score(Y_test,predict_heart_disease))

# Confusion Matrix 
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, predict_heart_disease))

# Detailed performance report
print("\nClassification Report:\n", classification_report(Y_test, predict_heart_disease))

#Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Count Plot(disease vs no disease)
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
plt.show()
