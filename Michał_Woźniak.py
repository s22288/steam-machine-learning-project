import  pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

pokemons =pd.read_csv('pokemon.csv',delimiter=',')
# print(pokemons.dtypes)
scaler = MinMaxScaler()
standardScaler = StandardScaler()
#  wartości obrony przed różnymi atakami
# abilities column =
pokemons = pokemons.drop('abilities',axis=1)

missing = [var for var in pokemons.columns if pokemons[var].isnull().sum()>0]
data_nan = pokemons[missing].isnull().mean()
data_nan = pd.DataFrame(data_nan.reset_index())
data_nan.columns = ['zmienna','procent']
data_nan.sort_values(by='procent',ascending=False,inplace=True)
# print(data_nan)

data_To_normalized = pokemons.columns[1:19]
pokemons[data_To_normalized] = pokemons[data_To_normalized].apply(lambda  x:(x - x.min()) / (x.max() - x.min()))
pokemons['attack'] = scaler.fit_transform(pokemons[['attack']])
mean_base_egg = round(pokemons['base_egg_steps'].mean())



pokemons['base_egg_steps'] = scaler.fit_transform(pokemons[['base_egg_steps']])
# usunięcie odstających wartośći z base happinsess
mean_happiness = round(pokemons['base_happiness'].mean())

# plt.matshow(pokemons.corr())
# plt.show()
# print((missing))
pokemons['base_happiness'] = pokemons['base_happiness'].apply(lambda  x:  mean_happiness if x>84 else x)
pokemons['base_happiness'] = scaler.fit_transform(pokemons[['base_happiness']])
# base total column
base_total_mean = pokemons['base_total'].mean()
pokemons['base_total'] = pokemons['base_total'].apply(lambda  x:  base_total_mean if x>720 else x)
pokemons['base_total'] = standardScaler.fit_transform(pokemons[['base_total']])
# capture rate
pokemons['capture_rate'] = pokemons['capture_rate'].apply(lambda x: str(x).isdigit())
pokemons['capture_rate'] = scaler.fit_transform(pokemons[['capture_rate']])
# classification columnn
pokemons = pokemons.drop('classfication',axis=1)
# denense column
defense_mean = pokemons['defense'].mean()
pokemons['defense'] = pokemons['defense'].apply(lambda  x:  defense_mean if x>163 else x)

pokemons['defense'] = scaler.fit_transform(pokemons[['defense']])
#experience growth
pokemons = pokemons.drop('experience_growth',axis=1)
#height_m column
mean_height = pokemons['height_m'].mean()
pokemons['height_m']=pokemons['height_m'].fillna(mean_height)

pokemons['height_m'] = scaler.fit_transform(pokemons[['height_m']])
mean_hp = pokemons['hp'].mean()
pokemons['hp'] = pokemons['hp'].apply(lambda  x:  defense_mean if x>154 else x)

pokemons['hp'] = scaler.fit_transform(pokemons[['hp']])
# hp column

# japanese_name column
pokemons = pokemons.drop('japanese_name',axis=1)
pokemons = pokemons.drop('name',axis=1)
pokemons = pokemons.drop('pokedex_number',axis=1)
# procentage male
mean_male = pokemons['percentage_male'].mean()
# tutaj
pokemons['percentage_male'].fillna(mean_male)


pokemons['percentage_male'] = pokemons['percentage_male'] / 100.0
# pokedex number
# sp attack column
attack_mean = pokemons['sp_defense'].mean()

pokemons['sp_attack'] = pokemons['sp_attack'].apply(lambda  x:  attack_mean if x>176 else x)
pokemons['sp_attack'] = standardScaler.fit_transform(pokemons[['sp_attack']])
# sp defencse column
defence_mean = pokemons['sp_defense'].mean()
pokemons['sp_defense'] = pokemons['sp_defense'].apply(lambda  x:  defence_mean if x>147 else x)
pokemons['sp_defense'] = standardScaler.fit_transform(pokemons[['sp_defense']])
# speed column
speed_mean = pokemons['speed'].mean()
pokemons['speed'] = pokemons['speed'].apply(lambda  x:  speed_mean if x>127 else x)
pokemons['speed'] = scaler.fit_transform(pokemons[['speed']])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# type 1
pokemons['type1'] = label_encoder.fit_transform(pokemons['type1'])

# type 2
pokemons = pokemons.drop('type2', axis=1)
# weight column
mean_weight = pokemons['weight_kg'].mean()
pokemons['weight_kg'] = pokemons['weight_kg'].fillna(mean_weight)
pokemons['weight_kg'] = pokemons['weight_kg'].apply(lambda  x:  mean_weight if x>401 else x)
# generation
pokemons['generation'] = scaler.fit_transform(pokemons[['generation']])
correlation_matrix = pokemons.corr()

legendary_correlations = correlation_matrix['is_legendary'].drop('is_legendary')

sorted_correlations = legendary_correlations.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index, palette='coolwarm')

plt.title('Korelacja cech z byciem legendarnym Pokémonem')
plt.xlabel('Współczynnik korelacji')
plt.ylabel('Cecha')

# Wyświetlenie wykresu
plt.show()
# próg korelacji
threshold = 0.3
positive_correlations = sorted_correlations[sorted_correlations > threshold]
negative_correlations = sorted_correlations[sorted_correlations < -threshold]

important_features = positive_correlations._append(negative_correlations)
print("Ważne cechy:")
print(important_features)

# tworzenie modeli
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# features = ['base_egg_steps', 'sp_attack','base_happiness', 'base_total','hp','height_m','attack','weight_kg']
features = ['against_dark', 'speed','against_poison', 'capture_rate','generation','type1','against_ghost','defense']

X = pokemons[features]
nan_counts = X.isna().sum()

# Wyświetlanie wyników
print(nan_counts)
y = pokemons['is_legendary']
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

# KNeighborsClassifier
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

# Visualization
# RandomForest feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_forest, y=feature_importance_forest.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for RandomForest Classifier')
plt.show()

# Knn feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance_svm, y=feature_importance_svm.index)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for SVM')
plt.show()
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
    'kNN': accuracy_knn,
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



