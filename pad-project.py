import matplotlib
import  pandas as pd
games = pd.read_csv('games.csv',delimiter=',')
games.set_index('AppID',inplace=True)
import re
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# usuniecie nie potrzebnych kolumn
games =games.drop('Name',axis=1)
games =games.drop('About the game',axis=1)
games =games.drop('Full audio languages',axis=1)
games =games.drop('Reviews',axis=1)
games =games.drop('Header image',axis=1)
games =games.drop('Website',axis=1)
games =games.drop('Support url',axis=1)
games =games.drop('Support email',axis=1)
games =games.drop('Metacritic score',axis=1)
games =games.drop('Metacritic url',axis=1)
games =games.drop('User score',axis=1)
games =games.drop('Notes',axis=1)
games =games.drop('Score rank',axis=1)
games =games.drop('Developers',axis=1)
games =games.drop('Publishers',axis=1)
games =games.drop('Tags',axis=1)
games =games.drop('Screenshots',axis=1)
games =games.drop('Movies',axis=1)
games =games.drop('Release date',axis=1)



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# games['Release date'] = scaler.fit_transform(games[['Release date']])

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()

games['Estimated owners'] = games['Estimated owners'].str.split('-',expand=True)[1]
games['Estimated owners'] =standardScaler.fit_transform(games[['Estimated owners']])

peak_mean =games['Peak CCU'].mean()
games['Peak CCU'] = games['Peak CCU'].apply(lambda  x:  peak_mean if x>403606 else x)

games['Peak CCU'] = standardScaler.fit_transform(games[['Peak CCU']])
# requred age
age_mean = games['Required age'].mean()
games['Required age'] = games['Required age'].apply(lambda  x:  age_mean if x>1.05 else x)

games['Required age'] =scaler.fit_transform(games[['Required age']] )
#Price
avg_price = games['Price'].mean()
games['Price'] = games['Price'].apply(lambda  x:  avg_price if x>50.0 else x)
games['Price'] = scaler.fit_transform(games[['Price']])

dlc_mean = games['DLC count'].mean()
games['DLC count'] = games['DLC count'].apply(lambda  x:  dlc_mean if x>119.0 else x)

games['DLC count'] =scaler.fit_transform(games[['DLC count']])


games['Supported languages'] =games['Supported languages'].apply(len)
games['Supported languages'] = scaler.fit_transform(games[['Supported languages']])
# metric score
# print(games)

# positibe and negative
positive_mean = games['Positive'].mean()
games['Positive'] = games['Positive'].apply(lambda  x:  positive_mean if x>288221.0 else x)

games['Positive'] = standardScaler.fit_transform(games[['Positive']])
negative_mean = games['Negative'].mean()

games['Negative'] = games['Negative'].apply(lambda  x:  negative_mean if x>44799.0 else x)

games['Negative'] = standardScaler.fit_transform(games[['Negative']])
# achivments
achivment_mean = games['Achievements'].mean()
games['Achievements'] = games['Achievements'].apply(lambda  x:  achivment_mean if x>492.0 else x)

games['Achievements'] = standardScaler.fit_transform(games[['Achievements']])

recommendation_mean = games['Recommendations'].mean()
games['Recommendations'] = games['Recommendations'].apply(lambda  x:  recommendation_mean if x>172079.0 else x)

games['Recommendations'] = standardScaler.fit_transform(games[['Recommendations']])
games['Average playtime forever'] = standardScaler.fit_transform(games[['Average playtime forever']])
games['Average playtime two weeks'] = standardScaler.fit_transform(games[['Average playtime two weeks']])
games['Median playtime forever'] = standardScaler.fit_transform(games[['Median playtime forever']])
games['Median playtime two weeks'] = standardScaler.fit_transform(games[['Median playtime two weeks']])
games['Median playtime forever'] = standardScaler.fit_transform(games[['Median playtime forever']])
# categories and genres

games['Categories'] = games['Categories'].fillna('')

games['Categories'] =(games['Categories'].str.split(',').apply(len))
games['Categories'] =scaler.fit_transform(games[['Categories']])


games['Genres'] = games['Genres'].fillna('')
games['Genres'] =(games['Genres'].str.split(',').apply(len))
games['Genres'] =scaler.fit_transform(games[['Genres']])
import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = games.corr()
linux_correlations = correlation_matrix['Linux'].drop('Linux')

sorted_correlations = linux_correlations.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, palette='coolwarm')

plt.title('Korelacja cech z Dzialaniem na systemie linux')
plt.xlabel('Współczynnik korelacji')
plt.ylabel('Cecha')

# Wyświetlenie wykresu
plt.show()
# próg korelacji
threshold = 0.1
positive_correlations = sorted_correlations[sorted_correlations > threshold]
negative_correlations = sorted_correlations[sorted_correlations < -threshold]

important_features = positive_correlations._append(negative_correlations)
print("Ważne cechy:")
print(important_features)


# Modelowanie
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

features = ['Mac', 'Categories', 'Achievements','Windows']

X = games[features]
y = games['Linux']
nan_counts = X.isna().sum()

# Wyświetlenie liczby brakujących wartości dla każdej cechy
print("Number of NaN values in each feature:")
print(nan_counts)

# Alternatywnie, można wyświetlić tylko te cechy, które zawierają NaN
features_with_nan = nan_counts[nan_counts > 0]
print("\nFeatures with NaN values:")
print(features_with_nan)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy (Random Forest): {accuracy_rf:.2f}')
print('Classification Report (Random Forest):')
print(classification_report(y_test, y_pred_rf))

importances_forest = clf_rf.feature_importances_
feature_importance_forest = pd.Series(importances_forest, index=features).sort_values(ascending=False)

KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy (kNN): {accuracy_knn:.2f}')
print('Classification Report (kNN):')
print(classification_report(y_test, y_pred_knn))

# SVM
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy (SVM): {accuracy_svm:.2f}')
print('Classification Report (SVM):')
print(classification_report(y_test, y_pred_svm))

coeff_svm = svm.coef_[0]
feature_importance_svm = pd.Series(coeff_svm, index=features).sort_values(ascending=False)

# Logistic Regression
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f'Accuracy (Logistic Regression): {accuracy_logreg:.2f}')
print('Classification Report (Logistic Regression):')
print(classification_report(y_test, y_pred_logreg))

coeff_logreg = logreg.coef_[0]
feature_importance_logreg = pd.Series(coeff_logreg, index=features).sort_values(ascending=False)
# with open('logreg.pkl', 'wb') as file:
#     pickle.dump(logreg, file)
#
# with open('knn.pkl', 'wb') as file:
#     pickle.dump(knn, file)
# with open('forest.pkl', 'wb') as file:
#     pickle.dump(clf_rf, file)
# with open('svm.pkl', 'wb') as file:
#     pickle.dump(svm, file)

# Visualization
# RandomForest feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_forest, y=feature_importance_forest.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for RandomForest Classifier')
plt.show()

# Knn feature importances
# plt.figure(figsize=(10,6))
# sns.barplot(x=feature_importance_svm, y=feature_importance_svm.index)
# plt.xlabel('Feature Importance')
# plt.ylabel('Features')
# plt.title('Feature Importance for SVM')
# plt.show()
# SVM feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_svm, y=feature_importance_svm.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for SVM')
plt.show()

# Logistic Regression feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_logreg, y=feature_importance_logreg.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Logistic Regression')
plt.show()

# Dokładności modeli
accuracies = {
    'Random Forest': accuracy_rf,
    # 'kNN': accuracy_knn,
    'SVM': accuracy_svm,
    'Logistic Regression': accuracy_logreg
}

# Wykres porównania dokładności modeli
plt.figure(figsize=(10,6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)
plt.show()


