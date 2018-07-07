import pandas as pd
from src.exploratory_data_analysis.exploratory_data_analysis import ExploratoryDataAnalysis
from src.model_fitting.Classifier import RFC
from src.preprocessing.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split


def preprocessing(current_dir, file_path):

    df = pd.read_csv(file_path)

    df_id = df['PassengerId']
    df = df.drop(['PassengerId'], axis=1)

    pp = Preprocessing()
    is_include, items = pp.is_include_category(df)
    df = pp.get_df_converted(df, items)
    missing = pp.get_missing_list(df)

    eda = ExploratoryDataAnalysis(current_dir, None)
    eda.show_statistics(df)

    df = pp.get_df_filled(df=df, missing_list=missing, method='median')

    return df, df_id


if __name__ == "__main__":

    def main():

        current_dir = 'src/prediction_pipeline/kaggle/titanic'
        data_dir = "C:/Users/yukiy/Documents/GitHub/machine_learning_codebase/data/kaggle/titanic/"
        train_data = data_dir + 'train.csv'
        test_data = data_dir + 'test.csv'

        df, df_id = preprocessing(current_dir, train_data)

        y = pd.DataFrame(df['Survived'])
        X = df.drop(y.columns, axis=1)
        y = y.as_matrix().ravel()

        random_state = 42

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        rfc = RFC(random_state)
        best_params = rfc.get_best_parameter(X=X, y=y, cv=5, n_iter=100, scoring='accuracy',
                                             n_jobs=7, verbose=0, random_state=random_state)
        rfc.set_hyper_parameter(best_params, random_state, 7)
        rfc.evaluate(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        df_test, df_id = preprocessing(current_dir, test_data)
        y_hat = rfc.get_df_y_hat(df_test=df_test, y_hat_name=['Survived'])
        df_output = pd.concat([df_id, y_hat], axis=1)
        df_output.to_csv(data_dir + 'df_output.csv', index=False)

    main()

