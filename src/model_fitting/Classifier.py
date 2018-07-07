import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


class RFC:

    rfc = RandomForestClassifier()

    def __init__(self, random_state):
        self.rfc = RandomForestClassifier(random_state=random_state)

    def get_best_parameter(self, X, y, cv, n_iter, scoring, n_jobs, verbose, random_state):

        n_features = len(X.columns)

        forest_param_dist = {
            'max_depth': [3, 5, 7],
            'n_estimators': [10, 20, 30, 40, 50],
            'max_features': sp_randint(1, n_features),
            'min_samples_split': sp_randint(2, n_features),
            'min_samples_leaf': sp_randint(1, n_features),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']}

        forest_random = RandomizedSearchCV(
            self.rfc,
            param_distributions=forest_param_dist,
            cv=cv,  # CV
            n_iter=n_iter,  # interation num
            scoring=scoring,  # metrics
            n_jobs=n_jobs,  # num of core
            verbose=verbose,
            random_state=random_state)

        forest_random.fit(X, y)
        return forest_random.best_params_

    def set_hyper_parameter(self, best_params, random_state, n_jobs):
        self.rfc = RandomForestClassifier(
            max_depth=best_params['max_depth'],
            n_estimators=best_params['n_estimators'],
            max_features=best_params['max_features'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            bootstrap=best_params['bootstrap'],
            criterion=best_params['criterion'],
            random_state=random_state, n_jobs=n_jobs)

    def evaluate(self, X_train, X_test, y_train, y_test):
        self.rfc.fit(X_train, y_train)
        ac = accuracy_score(y_test, self.rfc.predict(X_test))
        print('Accuracy is: ', ac)
        # 混同行列を出力する関数
        cm = confusion_matrix(y_test, self.rfc.predict(X_test))
        sns.set(style="whitegrid", palette="muted")
        sns.heatmap(cm, annot=True, fmt="d")
        plt.savefig('confusion_matrix.png')

    def get_df_y_hat(self, df_test, y_hat_name):
        df_y_hat = pd.DataFrame(self.rfc.predict(df_test))
        df_y_hat.columns = y_hat_name
        return df_y_hat
