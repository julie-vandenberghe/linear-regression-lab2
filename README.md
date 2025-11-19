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

L’EDA sert à comprendre les données avant de créer un modèle. Elle se divise en 3 grandes sous-parties :

- Analyse univariée numérique → comprendre chaque variable numérique individuellement
- Analyse bivariée numérique → comprendre la relation entre 2 variables numériques
- Corrélations / Heatmap → comprendre les relations globales entre toutes les variables

### Séparation des colonnes numériques et catégorielles

```python
df2_num = df2.select_dtypes(['int64', 'float64'])
df2_cat = df2.select_dtypes(['object'])
```

### Analyse univariée numérique

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

#### Scatter plots sur toutes les variables numériques

Dans notre exemple, on génére des graphiques de dispersion (scatter plots) afin de visualiser la relation entre chaque variable numérique et le prix de vente.
On voit ainsi que plus une voiture a roulé, moins elle vaut. Et plus la voiture était chère neuve, plus elle est chère d’occasion.

#### Scatter plot détaillé pour Kms_Driven

On fait ensuite un scatter plot détaillé pour Kms_Driven en dessous de 100000 kms car certaines voitures ont peut-être 300 000 ou même 600 000 km et ces valeurs sont beaucoup trop grandes et écrasent totalement l’échelle.

#### Heatmap de corrélation

Le but est ici de voir quelles variables numériques sont corrélées entre elles.
On voit ainsi que Present_Price corrèle fortement avec Selling_Price. Et que Age corrèle négativement avec Selling_Price.

-> A COMPLÈTER - JULIE

## 4. Modèle de régression linéaire

```python
df4
```

Permet d'afficher le DataFrame final `df4` qui contient les données prêtes pour la modélisation.

Il contient 299 lignes et 8 colonnes :

- Selling_Price
- Present_Price
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner
- Age

```python

scaler = MinMaxScaler(feature_range=(1,4))
norm = scaler.fit_transform(df4[['Present_Price', 'Kms_Driven','Age']])
norm = pd.DataFrame(norm, columns=['Present_Price', 'Kms_Driven','Age'])
norm = pd.concat([norm, df4[['Fuel_Type','Seller_Type',	'Transmission',	'Owner', 'Selling_Price']]], axis=1)
norm
```

Ce code normalise les colonnes 'Present_Price', 'Kms_Driven' et 'Age' du DataFrame `df4` en utilisant la méthode Min-Max Scaling pour les faire varier entre 1 et 4. Ensuite, il crée un nouveau DataFrame `norm` qui combine les colonnes normalisées avec les autres colonnes non normalisées du DataFrame original.

La fonction `MinMaxScaler` de `sklearn.preprocessing` est utilisée pour effectuer cette normalisation en étirant les valeurs des colonnes sélectionnées dans la plage spécifiée (1 à 4 dans ce cas).

```python

%matplotlib inline

CV = []
R2_train = []
R2_test = []
MAE_train = []
MAE_test = []

class CarPredModel:
    def __init__(self, x, y, test_size=0.3):
        """get x, y and test_size from user
            x(Dataframe): features
            y(Dataframe_1d): target
            test_size(float): for train test split
        """
        self.x = x
        self.y = y
        self.test_size = test_size


    def fit_model(self):
        """
        fit model on x and y
        claculate R2, MSE and MAE for train and test
        """
        # train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=0)

        # fit model
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)

        # R2 Score of train set:
        self.y_pred_train = self.model.predict(self.x_train)
        self.r2_train_model = metrics.r2_score(self.y_train, self.y_pred_train)
        # MAE and MSE of train set:
        self.mae_train_model = metrics.mean_absolute_error(self.y_train, self.y_pred_train)

        # R2 Score of test set:
        self.y_pred_test = self.model.predict(self.x_test)
        self.r2_test_model = metrics.r2_score(self.y_test, self.y_pred_test)
        # MAE and MSE of train set:
        self.mae_test_model = metrics.mean_absolute_error(self.y_test, self.y_pred_test)
        self.mse_test_model = metrics.mean_squared_error(self.y_test, self.y_pred_test)


    def cross(self, k):
        """
        Perform cross validation
        printing result of model
        """
        # R2 mean of train set using Cross validation:
        kf = KFold(k)
        self.cross_val = cross_val_score(self.model, self.x_train, self.y_train, cv=kf, scoring='r2')
        self.cv_mean = np.mean(self.cross_val)


        # Printing results
        print('='*30,'Shape','='*30)
        print("x train: ",self.x_train.shape)
        print("x test: ",self.x_test.shape)
        print("y train: ",self.y_train.shape)
        print("y test: ",self.y_test.shape)
        print('='*30,'R2_score and CV','='*30)
        print("Train R2-score :", round(self.r2_train_model, 3))
        print("Test R2-score :", round(self.r2_test_model, 3))
        print("Train MAE :", round(self.mae_train_model, 3))
        print("Test MAE :", round(self.mae_test_model, 3))
        print("Train CV scores :", self.cross_val)
        print("Train CV mean :", round(self.cv_mean, 3))

    def plot_graph(self):
        """
        plotting the result
        """
        # Plotting Graphs
        # Residual Plot of train data
        fig, ax = plt.subplots(1,3,figsize = (15,4))
        ax[0].set_title('Residual Plot of Train samples', fontsize=14, fontweight='bold')
        sns.histplot((self.y_train-self.y_pred_train), kde=True, ax = ax[0])
        ax[0].set_xlabel('y_train - y_pred_train')
        # Y_test vs Y_pred_test scatter plot
        ax[1].set_title('y_test vs y_pred_test', fontsize=14, fontweight='bold')
        ax[1].scatter(x = self.y_test, y = self.y_pred_test)
        ax[1].set_xlabel('y_test')
        ax[1].set_ylabel('y_pred_test')
        # MAE_test vs MAE_train line plot
        ax[2].set_title('MAE test vs MAE train', fontsize=14, fontweight='bold')
        sns.lineplot(data=pd.DataFrame({'MAE_train': MAE_train, 'MAE_test': MAE_test}), markers=True)
        ax[2].set_xlabel('M')
        ax[2].set_ylabel('MAE')

        plt.show()

        # Print results of model again
        print("Train R2-score :", round(self.r2_train_model, 3))
        print("Test R2-score :", round(self.r2_test_model, 3))

    def append_result(self):
        """store R2, MAE and CV"""
        R2_train.append(round(self.r2_train_model, 3))
        MAE_train.append(round(self.mae_train_model, 3))
        R2_test.append(round(self.r2_test_model, 3))
        MAE_test.append(round(self.mae_test_model, 3))
        CV.append(round(self.cv_mean, 3))

    def show_weight(self):
        """create table of coef and intercept of model"""
        # The parameters for linear regression model
        parameter = ['b']+ ['w_' + str(i) for i in range(1,self.x.shape[1]+1)]
        columns = ['intercept'] + self.x.columns.to_list()
        weight_table = pd.DataFrame({'Parameter': parameter, 'Columns': columns})
        sk_weight = [i for i in self.model.intercept_] + self.model.coef_.tolist()[0]
        weight_table = weight_table.join(pd.Series(sk_weight, name='Sk_weight'))
        return weight_table
```

Ce code définit une classe `CarPredModel` qui encapsule le processus de création, d'entraînement, d'évaluation et de visualisation d'un modèle de régression linéaire pour prédire le prix de vente des voitures.

La fonction `__init__` initialise la classe avec les caractéristiques (x), la cible (y) et la taille du test.

La fonction `fit_model` divise les données en ensembles d'entraînement et de test, ajuste le modèle de régression linéaire, et calcule les scores R2 et MAE pour les ensembles d'entraînement et de test.\
La fonction utilise la méthode `train_test_split` de `sklearn.model_selection` pour diviser les données. Ensuite elle instantie un modèle de régression linéaire à l'aide de `LinearRegression` de `sklearn.linear_model`, ajuste le modèle avec les données d'entraînement, et prédit les valeurs pour les ensembles d'entraînement et de test. Enfin, elle calcule les scores R2, MAE et MSE en utilisant les fonctions de `sklearn.metrics`.

- score R2 (coefficient de détermination) : mesure la proportion de la variance dans la variable dépendante qui est prévisible à partir des variables indépendantes.
- MAE (Mean Absolute Error) : mesure la moyenne des erreurs absolues entre les valeurs prédites et les valeurs réelles.
- MSE (Mean Squared Error) : mesure la moyenne des carrés des erreurs entre les valeurs prédites et les valeurs réelles.

La fonction `cross` effectue une validation croisée k-fold pour évaluer la performance du modèle de manière plus robuste. Autrement dit, elle divise les données d'entraînement en k sous-ensembles, ajuste le modèle k fois en utilisant un sous-ensemble différent comme ensemble de validation à chaque fois, et calcule la moyenne des scores R2 obtenus.

La fonction `plot_graph` crée des graphiques pour visualiser les résidus, la relation entre les valeurs réelles et prédites, et les erreurs absolues moyennes.

La fonction `append_result` stocke les scores R2, MAE et CV dans des listes pour une analyse ultérieure.

La fonction `show_weight` crée un tableau des coefficients et de l'interception du modèle de régression linéaire.

Par la suite, on prépare les données pour le modèle :

```python
norm1 = norm.copy()
x = norm1.drop('Selling_Price', axis='columns')
y = norm1.Selling_Price.values.reshape(-1,1)
```

Ce code crée une copie du DataFrame `norm` appelée `norm1`, puis sépare les caractéristiques (x) en supprimant la colonne 'Selling_Price' et stocke la variable cible (y) en extrayant la colonne 'Selling_Price' et en la remodelant en un tableau 2D.

Enfin, on crée une instance de la classe `CarPredModel`, on ajuste le modèle et on effectue une validation croisée :

```python
model = CarPredModel(x, y, test_size=0.2)
model.fit_model()
model.cross(k=5)
```

Ce code crée une instance de la classe `CarPredModel` en utilisant les caractéristiques (x) et la cible (y) avec une taille de test de 20%. Ensuite, il ajuste le modèle en appelant la méthode `fit_model()` et effectue une validation croisée à 5 plis en appelant la méthode `cross(k=5)`.

`model1.cross(10)` effectue une validation croisée à 10 plis sur le modèle `model1`, ce qui permet d'évaluer la performance du modèle de manière plus robuste en utilisant 10 sous-ensembles différents des données d'entraînement.

```python
model1.append_result()
model1.plot_graph()
model1.show_weight()
```

Ce code appelle la méthode `append_result()` pour stocker les résultats du modèle, puis appelle la méthode `plot_graph()` pour visualiser les performances du modèle à l'aide de graphiques, et enfin appelle la méthode `show_weight()` pour afficher les coefficients et l'interception du modèle de régression linéaire.

## 5. Amélioration du modèle

-> AURELIE

## 6. Visualisation du modèle final

## 7. Prédictions de données simples

### Sélection du modèle optimal

Après comparaison des quatre modèles développés (model1, model2, model3, model4), le **model4** a été retenu car il présente les meilleures performances avec un **coefficient de détermination R² = 0.97**. Ce score signifie que le modèle explique 97% de la variance des prix de voitures, ne laissant que 3% de variabilité inexpliquée.

### Système de prédiction interactif

1. **Saisie des données** : L'utilisateur peut entrer les caractéristiques d'une voiture :

   - Prix actuel (Present_Price) en lakhs
   - Âge de la voiture en années
   - Kilométrage parcouru (Kms_Driven)
   - Type de vendeur : Dealer (2) ou Individual (3)
   - Type de carburant : Diesel (2), Petrol (3), CNG (4)
   - Type de transmission : Automatic (2) ou Manual (3)
   - Nombre de propriétaires précédents

2. **Prétraitement automatique** : Le système applique automatiquement :

   - L'encodage des variables catégorielles
   - La normalisation des variables numériques (MinMaxScaler)
   - Le feature engineering (log transformations, interactions)
   - La création des nouvelles features : `log_Age`, `log_Kms_Driven`, `p_price_log_age`, `p_price_fuel`

### Implémentation technique détaillée

#### 1. Préparation des données initiales

```python
x = df4.drop('Selling_Price', axis=1)
```

- Creer le dataframe, supprime la coline selling price

```python
y = df4.Selling_Price.values.reshape(-1,1)
```

- Créer la variable cible transforme en matrice

#### 2. Configuration de l'interface utilisateur

```python
cols_name = ['Present_Price','Age','Kms_Driven','Seller_Type','Fuel_Type','Transmission','Owner']
```

- Définir la liste des caractéristiques à saisir par l'utilisateur

#### 3. Saisie interactive des données

```python
sample = pd.DataFrame()
```

- Créer un DataFrame vide pour stocker la nouvelle observation

```python
for col in cols_name:
    sample[col] = [float(input(f"{col}: "))]
```

- Met les valeurs que l'utilisateur à entrer pour chaque caracterique

#### 4. Intégration et Feature Engineering

```python
x = pd.concat([x,sample])
```

- Ajoute la nouvelle observation aux données existantes, --> concat = empiler les lignes

```python
x['log_Age'] = np.log10(x['Age'])
```

- Créer une nouvelle colonne avec la transformation logarithmique de l'âge

```python
x['log_Kms_Driven'] = np.log10(x['Kms_Driven'])
```

- Transformation logarithmique du kilométrage
- log pcq : Réduit l'asymétrie et améliore la linéarité

```python
x['p_price_log_age'] = x['Present_Price'] * x['log_Age']
```

- Comment prix et âge interagissent ensemble

```python
x = x.drop(['Age', 'Kms_Driven'], axis=1)
```

- Supprime les variables originales (Age, Kms_Driven) pcq on utilise leurs versions logarithmiques
- Évite la redondance

#### 5. Division des données

```python
x_train = x[:len(x)-1]
```

- Créer les données d'entraînement

```python
x_test = x[len(x)-1:]
```

- Créer les données de test

```python
y_train = y
```

- Variables cibles pour l'entraînement

#### 6. Entraînement du modèle

```python
final_model = LinearRegression()
```

- Créer une instance de régression linéaire vierge

```python
final_model.fit(x_train, y_train)
```

- Entraîner le modèle sur les données

#### 7. Analyse des coefficients

```python
final_params = ['b']+ ['w_' + str(i) for i in range(1,x.shape[1]+1)]
```

- Créer les noms des paramètres

```python
param_name = ['intercept'] + x.columns.to_list()
```

- Créer la liste des noms de colonnes

```python
final_weight_table = pd.DataFrame({'final_params': final_params, 'Columns': param_name})
```

- Créer un DataFrame pour le tableau des coefficients

```python
sk_weight = [i for i in final_model.intercept_] + final_model.coef_.tolist()[0]
```

- Extraire tous les coefficients du modèle

```python
final_weight_table = final_weight_table.join(pd.Series(sk_weight, name='Sk_weight'))
```

- Ajouter les valeurs des coefficients au tableau
- pd.Series() : Crée une série avec les poids

#### 8. Prédiction finale

```python
y_pred = final_model.predict(x_test)
```

- Prédire le prix de la nouvelle voiture

### Avantages de l'approche

1. **Robustesse du modèle**
2. **Facilité d'utilisation**
3. **Transparence et traçabilité**
