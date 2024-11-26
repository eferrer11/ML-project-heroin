from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import main as m

"""
data = pd.read_csv("./train.csv")
spectrum = data.iloc[:, 6:]
spectrum_filtered = pd.DataFrame(savgol_filter(spectrum, 7, 3, deriv = 2, axis = 0))
spectrum_filtered_st = zscore(spectrum_filtered, axis = 1)
"""

"""
test_data = pd.read_csv("./test.csv")
spectrum_test = test_data.iloc[:, 6:]
spectrum_test_filtered = pd.DataFrame(savgol_filter(spectrum_test, 7, 3, deriv = 2, axis = 0))
spectrum_test_filtered_st = zscore(spectrum_test_filtered, axis=1)
"""

X = m.spectrum_filtered_st
y = m.data['PURITY']
X_train, X_valid, y_train , y_valid = train_test_split(X, y, test_size=0.05, random_state=42)


def fit_and_evaluate(K:int, x_train:pd.DataFrame, x_valid:pd.DataFrame, Y_train, Y_valid)->dict:
    model = KNeighborsRegressor(n_neighbors=K)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    return {
        "training_error": np.sqrt(mean_squared_error(model.predict(x_train), Y_train)),
        "test_error": np.sqrt(mean_squared_error(model.predict(x_valid), Y_valid)),
        "prediction": y_pred,
    }

KNN = fit_and_evaluate(40, X_train, X_valid, y_train, y_valid)
#print(KNN)


# cross-validation function
def cross_validation(idxs, nfolds=5):
    n = int(np.ceil(len(idxs) / nfolds))
    validation_sets = [idxs[k * n : min((k + 1) * n, len(idxs))] for k in range(nfolds)]
    train_sets = [list(set(idxs) - set(validation_set)) for validation_set in validation_sets]
    return {'train_sets': train_sets, 'validation_sets': validation_sets}

def nested_cross_validation(idxs, inner_nfolds=5, outer_nfolds=5):
    outer = cross_validation(idxs, nfolds=outer_nfolds)
    results = []

    # Parcourir les folds externes
    for te, tr in zip(outer['validation_sets'], outer['train_sets']):
        
        # Validation croisée interne pour optimiser les hyperparamètres
        inner = cross_validation(tr, nfolds=inner_nfolds)
        
        best_score = float('inf')  # Initialiser avec un score très élevé
        best_k = None  # Initialiser le meilleur k à None

        # Essayer différentes valeurs de K dans la validation croisée interne
        k_range = range(10, 40)  # Tester les valeurs de K de 1 à 20
        for k in k_range:
            score = 0  # Calculer la moyenne des scores pour ce k
            
            for inner_train, inner_test in zip(inner['train_sets'], inner['validation_sets']):
                X_train, X_val = X.iloc[inner_train], X.iloc[inner_test]
                y_train, y_val = y.iloc[inner_train], y.iloc[inner_test]
                
                model = KNeighborsRegressor(n_neighbors=k)
                model.fit(X_train, y_train)
                score += np.sqrt(mean_squared_error(y_val, model.predict(X_val)))  # Accumuler les scores
                
            avg_score = score / inner_nfolds  # Moyenne des scores pour ce K
            
            # Si ce K donne un meilleur score, on met à jour best_score et best_k
            if avg_score < best_score:
                best_score = avg_score
                best_k = k

        # Evaluation sur le fold externe avec le meilleur K trouvé
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        
        model = KNeighborsRegressor(n_neighbors=best_k)  # Utiliser le meilleur K
        model.fit(X_train, y_train)
        test_error = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        # Sauvegarder les résultats de ce fold externe
        results.append({
            'test_set': te,
            'best_k': best_k,
            'best_score': best_score,
            'test_error': test_error
        })
    
    return results

# Exemple d'utilisation
idxs = list(range(len(X_train)))  # Indices de vos données d'entraînement
results = nested_cross_validation(idxs, inner_nfolds=5, outer_nfolds=5)

# Affichage des résultats
for result in results:
    print(f"Best K: {result['best_k']}")
    print(f"Best internal validation score (MSE): {result['best_score']}")
    print(f"Test error (MSE): {result['test_error']}")
    print("-" * 50)

t_score = np.mean(np.abs(KNN["prediction"] - y_valid) <= 5)
print(t_score)