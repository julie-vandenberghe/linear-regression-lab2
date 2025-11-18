---
title: "Analyse de Régression Linéaire sur les Prix de Voitures"
author: "Satya Dejonge, Aurélie Dulymbois, Audrey Hertaux-Sene, Julie Vandenberghe, Clément Szewczyk"
date: "Novembre 2025"
promo: "M1 Cyber"
---

# Analyse du modèle de régression linéaire car_price_prediction_oop_tk.ipynb


## 0. Problématique

On veut prédire le prix de vente (Selling_Price) d’une voiture d’occasion à partir d’attributs comme : le modèle, l’année, le prix présent, les kilomètres parcourus, le type de carburant, type de vendeur, transmission, etc.

C’est un problème de régression : la variable cible est numérique continue (prix).

## 1. Librairies importées

### Visualisation et manipulation des données

- `Pandas` pour la manipulation des données
- `NumPy` pour les opérations numériques
- `Matplotlib` pour la visualisation des données
- `Seaborn` pour des graphiques statistiques avancés
- `plotly.express` pour des visualisations interactives
- `plotly.io` pour l'affichage des graphiques interactifs
- `plotly.graph_objects` pour des graphiques personnalisés
- `plt.style.use('_mpl-gallery')` pour définir le style des graphiques Matplotlib

### Apprentissage automatique

- `sklearn.model_selection` pour la division des données en ensembles d'entraînement et de test
- `sklearn.preprocessing` pour la normalisation et le prétraitement des données
- `sklearn.linear_model` pour la régression linéaire
- `sklearn.model_selection` pour la validation croisée
- `sklearn.metrics` pour l'évaluation des modèles

## 2. Chargement, copie et nettoyage des données

### Informations sur le CSV

Le CSV comporte 301 lignes et 9 colonnes.

**Les différentes colonnes sont :**

- Car_Name
- Year
- Selling_Price
- Present_Price
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner

### Pré-traitement des données

`df.drop(['Car_Name'], axis=1, inplace=True)` est utilisé pour supprimer la colonne 'Car_Name' du DataFrame, car elle n'est pas nécessaire pour l'analyse.

### Types de données

- Year : Entier
- Selling_Price : Flottant
- Present_Price : Flottant
- Kms_Driven : Entier
- Fuel_Type : Objet (chaîne de caractères)
- Seller_Type : Objet (chaîne de caractères)
- Transmission : Objet (chaîne de caractères)
- Owner : Entier

Il y a donc 2 colonnes avec des floats, 3 colonnes avec des entiers et 3 colonnes avec des objets (chaînes de caractères).

### Nettoyage des données

`df.isna().sum()` est utilisé pour vérifier la présence de valeurs manquantes dans chaque colonne du DataFrame. Si le résultat montre des zéros pour toutes les colonnes, cela signifie qu'il n'y a pas de valeurs manquantes dans le jeu de données.

`df.describe()` fournit des statistiques descriptives pour les colonnes numériques du DataFrame, telles que la moyenne, l'écart type, les valeurs minimales et maximales, ainsi que les quartiles. Cela aide à comprendre la distribution des données et à identifier d'éventuelles anomalies ou valeurs aberrantes.

`df.describe(include='all')` fournit des statistiques descriptives pour toutes les colonnes du DataFrame, y compris les colonnes non numériques. Cela inclut des informations telles que le nombre de valeurs uniques, la valeur la plus fréquente (mode) et la fréquence de cette valeur pour les colonnes de type objet (chaîne de caractères).

`df_cat= df.select_dtypes(['object'])` est utilisé pour sélectionner toutes les colonnes du DataFrame qui sont de type objet (chaîne de caractères) et les stocker dans un nouveau DataFrame appelé `df_cat`. Cela permet de se concentrer sur l'analyse des variables catégorielles.

`df_num= df.select_dtypes(['int64','float64'])` est utilisé pour sélectionner toutes les colonnes du DataFrame qui sont de type entier (int64) ou flottant (float64) et les stocker dans un nouveau DataFrame appelé `df_num`. Cela permet de se concentrer sur l'analyse des variables numériques.

### Visualisation des données

```python

for col in df_cat.columns:
    check_syntax = df_cat[col].value_counts().reset_index()
    check_syntax.columns = [col, 'Count']
    ax = sns.barplot(data=check_syntax, x=col, y='Count')
    ax.bar_label(ax.containers[0])
    plt.title(f"Count {col} column", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()
```

Ce code crée des graphiques à barres pour chaque colonne catégorielle dans le DataFrame `df_cat`, affichant le nombre d'occurrences de chaque catégorie.

- Graphique 1 : Répartition des types de carburant (Fuel_Type)
- Graphique 2 : Répartition des types de vendeurs (Seller_Type)
- Graphique 3 : Répartition des types de transmission (Transmission)

```python
df_cat.Fuel_Type.value_counts().plot.barh()
```
Ce code crée un graphique à barres horizontales pour la répartition des types de carburant (Fuel_Type) dans le DataFrame `df_cat`.

```python
ax = df_cat.Fuel_Type.value_counts().plot.barh()
ax.bar_label(ax.containers[0])
plt.title("Fuel_Type Distribution", fontsize=14, fontweight='bold')
plt.show()
```
Ce code crée un graphique à barres horizontales pour la répartition des types de carburant (Fuel_Type) dans le DataFrame `df_cat`, en ajoutant des étiquettes aux barres pour indiquer le nombre d'occurrences de chaque catégorie.

```python
ax = df_cat.Transmission.value_counts().plot.barh()
ax.bar_label(ax.containers[0])
plt.title("Transmission Distribution", fontsize=14, fontweight='bold')
plt.show()
```

Ce code crée un graphique à barres horizontales pour la répartition des types de transmission (Transmission) dans le DataFrame `df_cat`, en ajoutant des étiquettes aux barres pour indiquer le nombre d'occurrences de chaque catégorie.

```python
ax = df_cat.Seller_Type.value_counts().plot.barh()
ax.bar_label(ax.containers[0])
plt.title("Seller_Type Distribution", fontsize=14, fontweight='bold')
plt.show()
```

Ce code crée un graphique à barres horizontales pour la répartition des types de vendeurs (Seller_Type) dans le DataFrame `df_cat`, en ajoutant des étiquettes aux barres pour indiquer le nombre d'occurrences de chaque catégorie.

`df.duplicated().sum()` est utilisé pour vérifier la présence de lignes dupliquées dans le DataFrame. Si le résultat est zéro, cela signifie qu'il n'y a pas de lignes dupliquées dans le jeu de données.\
Ici, le résultat est `np.int64(2)`, ce qui indique qu'il y a 2 lignes dupliquées dans le DataFrame.

`df[df.duplicated()]` est utilisé pour afficher les lignes dupliquées dans le DataFrame. Cela permet d'examiner les données dupliquées et de décider comment les gérer (par exemple, les supprimer ou les conserver).

Les lignes dupliquées ont été supprimées en utilisant `df.drop_duplicates(inplace=True)`, ce qui modifie le DataFrame en place pour éliminer les doublons.

``df = df.reset_index(drop=True)`` est utilisé pour réinitialiser les index du DataFrame après la suppression des lignes dupliquées. L'argument `drop=True` indique que l'ancien index ne doit pas être ajouté comme une colonne dans le DataFrame.

```python
df2 = df.copy()
df2['Age'] = (df.Year.max()+1) - df.Year
df2.drop(['Year'], axis='columns', inplace=True)
df2
```

Ce code crée une copie du DataFrame `df` appelée `df2`, calcule l'âge des voitures en soustrayant l'année de chaque voiture de l'année maximale plus un, ajoute cette nouvelle colonne 'Age' à `df2`, puis supprime la colonne 'Year' originale.

```python
%matplotlib inline
for col in df2[['Present_Price', 'Kms_Driven', 'Owner', 'Age']]:
    sns.scatterplot(data=df2, x=col, y=df2.Selling_Price)
    plt.show()
```

Ce code crée des graphiques de dispersion pour chaque colonne numérique dans le DataFrame `df2` par rapport à la colonne 'Selling_Price', permettant de visualiser les relations entre ces variables.

- Graphique 1 : Relation entre le prix de vente (Selling_Price) et le prix actuel (Present_Price)
- Graphique 2 : Relation entre le prix de vente (Selling_Price) et les kilomètres parcourus (Kms_Driven)
- Graphique 3 : Relation entre le prix de vente (Selling_Price) et le nombre de propriétaires précédents (Owner)
- Graphique 4 : Relation entre le prix de vente (Selling_Price) et l'âge de la voiture

## 3. Exploratory Data Analysis (EDA)

### Séparation des colonnes numériques et catégorielles

```python
df2_num = df2.select_dtypes(['int64', 'float64'])
df2_cat = df2.select_dtypes(['object'])
```

On fait ensuite un `describe()` sur les données catégorielles et numériques.
Sur un DataFrame catégoriel (object), cela donne un résumé statistique (en termes de fréquence et de diversité) des colonnes catégorielles :
- count → nombre de valeurs non nulles
- unique → nombre de catégories distinctes
- top → catégorie la plus fréquente
- freq → fréquence de cette catégorie la plus fréquente

`.T` transpose ensuite le résultat pour que les colonnes deviennent des lignes (plus lisible).

On fait également un `describe()` sur les colonnes numériques (comme nous l'avions fait plus haut).

### Analyse bivariée numérique

L'analyse bivariée numérique est une analyse statistique ou graphique qui examine la relation entre deux variables numériques. 
On génére tout d'abord des graphiques de dispersion afin de visualiser la relation entre chaque variable numérique et le prix de vente.

[A COMPLÈTER]

## 4. Modèle de régression linéaire