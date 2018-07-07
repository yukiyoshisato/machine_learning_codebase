import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ExploratoryDataAnalysis:

    """
    Class for exploratory data analysis
    """

    def __init__(self, current_dir, logger):
        self.current_dir = current_dir
        self.logger = logger

    @classmethod
    def show_statistics(cls, df: pd.DataFrame):
        print(df.describe(include='all'))


# check NaN
    @classmethod
    def draw_std(cls, df: pd.DataFrame):
        df.plot()

    @classmethod
    def plot_correlation_map(cls, X, y):
        sns.set(style="whitegrid", palette="muted")
        data_dia = y
        data = X
        data_n_2 = (data - data.mean()) / (data.std())  # standardization
        data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
        data = pd.melt(data, id_vars="Survived",
                       var_name="features",
                       value_name='value')
        plt.figure(figsize=(10, 10))
        sns.swarmplot(x="features", y="value", hue="Survived", data=data)

        plt.xticks(rotation=90)
        plt.show()

        # correlation map
        f, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.show()

# check categorical variables and show patterns

# show box plot

# show violin plot

# show correlation map

