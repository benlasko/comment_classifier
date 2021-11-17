from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def fit_and_score_model(model, X_train, y_train, X_test, y_test, weighted=True):
    '''
    Fits train/test data to a model and prints accuracy, precision, recall, and f1 scores.

    Parameter
    ----------
    model:  model object
        A model to fit and score.

    X_train:  arr
        X_train data.

    y_train:  arr
        y_train data.

    X_test:  arr
        X_test data.

    y_test:  arr
        y_test data.

    weighted:  boolean
        If True then for precision, recall and f1 the average='weighted'.  Default = True.

    Returns
    ----------
    None.
    '''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    if weighted==True:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')


def grid_searcher(model, param_grid, X_train, y_train, cv=5):
    '''
    Performs grid search and returns best parameters.

    Parameters
    ----------
    model:  model object
        The model to grid search.

    X_train:  arr
        X_train data.

    y_train:  arr
        y_train data.

    param_grid:  dict
        Dictionary of parameters for grid search to test.

    cv:  int
        Number of folds for cross validation.  Default = 5.

    Returns
    ----------
    Best parameters found by the grid search for the model.
    '''
    mod_grid_search = GridSearchCV(model, param_grid, cv=cv)
    mod_grid_search.fit(X_train, y_train)
    return mod_grid_search.best_params_
