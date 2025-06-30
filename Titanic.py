import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

titanic_data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

titanic_data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

label_encoders = {}
for column in ['Sex', 'Embarked']:
    encoder = LabelEncoder()
    titanic_data[column] = encoder.fit_transform(titanic_data[column])
    label_encoders[column] = encoder

features = titanic_data.drop('Survived', axis=1)
target = titanic_data['Survived']

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(features_train, target_train)

predictions = classifier.predict(features_test)

print("Accuracy:", accuracy_score(target_test, predictions))
print("Classification Report:\n", classification_report(target_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(target_test, predictions))

