import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

file_path = './nba2021.csv'
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(['Player', 'Tm', 'Age', 'FG','FGA','3P','3PA','2P','2PA','FT','FTA' ,'PTS', 'DRB', 'ORB', 'GS','G'], axis=1)
#print(df.columns)

# Split the data into features and labels
X = df.drop('Pos', axis=1)
y = df['Pos']

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Initialize and train the Random Forest classifier
rf_classifier = MLPClassifier(activation="identity",solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(100,))
rf_classifier.fit(X_train, y_train)

print("Training set score: {:.3f}".format(rf_classifier.score(X_train, y_train)))
print("Test set score: {:.3f}".format(rf_classifier.score(X_test,y_test)))

prediction = rf_classifier.predict(X_test)
print("Confusion matrix:")
print(pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

cross_val_results = cross_val_score(rf_classifier, X, y, cv=10)  # 10-fold cross-validation
print('Cross-validation results:', cross_val_results)
print('Mean accuracy across folds: {:.3f}'.format(cross_val_results.mean()))





