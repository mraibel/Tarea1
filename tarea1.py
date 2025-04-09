import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./student_depression_dataset.csv')
df

# Preprocesamiento de datos
df.drop('id',axis=1,inplace=True) # Elimina el id
df = pd.get_dummies(df, columns=['Gender'], drop_first=True) # Transforma Male 1 y Female 0

# Codifica las ciudades en numeros enteros
le = LabelEncoder()
a = le.fit_transform(df['City'])
df['City'] = a

# Codifica las profesiones en numeros enteros
le1 = LabelEncoder()
b = le1.fit_transform(df['Profession'])
df['Profession'] = b

others = df['Sleep Duration'] == 'Others'
df.loc[others,'Sleep Duration'] = '6-7 hours'
le2 = LabelEncoder()
c =  le2.fit_transform(df['Sleep Duration'])
df['Sleep Duration'] = c

le3 = LabelEncoder()
d =  le3.fit_transform(df['Dietary Habits'])
df['Dietary Habits'] = d

le4 = LabelEncoder()
e =  le4.fit_transform(df['Degree'])
df['Degree'] = e

df = pd.get_dummies(df,columns = ['Have you ever had suicidal thoughts ?','Family History of Mental Illness'],drop_first=True)

df.drop(df[df['Financial Stress']=='?'].index,axis=0,inplace=True)
df['Financial Stress']=pd.to_numeric(df['Financial Stress'])

df

# División de la data y target
X = df.drop('Depression',axis=1) # Data
y = df['Depression'] # Target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,train_size=0.3)

# Random Forest

# Entrenamiento modelo de Random Forest
rfc= RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred_RF = rfc.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_RF)
print(f"Precisión del modelo: {accuracy:.2f}")
# Ver un informe de clasificación más detallado
report = classification_report(y_test, y_pred_RF)
print("Informe de clasificación:\n", report)

# 7. Mostrar la matriz de confusión
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_RF))
print(classification_report(y_test, y_pred_RF))

# **Visualización de la matriz de confusión para modelo de Random Forest**
cm_bin = confusion_matrix(y_test, y_pred_RF)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title('Matriz de Confusión - Random Forest')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Regresión logística


LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

y_pred_LR = LR_model.predict(X_test)

# 7. Mostrar la matriz de confusión
print("Regresión logística:")
print(confusion_matrix(y_test, y_pred_LR))
print(classification_report(y_test, y_pred_LR))

# **Visualización de la matriz de confusión para modelo de Regresión logística**
cm_bin = confusion_matrix(y_test, y_pred_LR)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title('Matriz de Confusión - Regresión logística')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# KNN (k-Nearest Neighbors)


# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo con k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo (realmente solo guarda los datos)
knn.fit(X_train_scaled, y_train)

# Predecir
y_pred_KNN = knn.predict(X_test_scaled)

# 7. Mostrar la matriz de confusión
print("KNN:")
print(confusion_matrix(y_test, y_pred_KNN))
print(classification_report(y_test, y_pred_KNN))

# **Visualización de la matriz de confusión para modelo de KNN**
cm_bin = confusion_matrix(y_test, y_pred_KNN)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 0', 'Clase 1'], yticklabels=['Clase 0', 'Clase 1'])
plt.title('Matriz de Confusión - KNN')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Comparación de modelos con métricas

# Calcular métricas para cada modelo
models = {
    "Random Forest": y_pred_RF,
    "Regresión Logística": y_pred_LR,
    "KNN": y_pred_KNN
}

# Crear listas para almacenar las métricas
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Llenar las listas con métricas para cada modelo
for name, y_pred in models.items():
    model_names.append(name)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# Crear un DataFrame con los resultados
comparison_df = pd.DataFrame({
    "Modelo": model_names,
    "Accuracy": accuracies,
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1_scores
})

# Mostrar la tabla
print("Comparación de Modelos:")
print(comparison_df.sort_values(by="F1-Score", ascending=False))