import pandas as pd


class Preprocessing:

    @classmethod
    def is_include_category(cls, df: pd.DataFrame):
        """
        If df includes category label, return True and labels
        :param df:
        :return:
        """
        is_include = False
        items = list([])
        for column, item in df.iteritems():
            if item.dtypes == 'object':
                is_include = True
                items.append(column)
        return is_include, items

    @classmethod
    def get_df_converted(cls, df: pd.DataFrame, items: list):

        drop_items = list([])
        for item in items:
            if float(len(df[item].drop_duplicates())) / len(df) > 0.01:
                drop_items.append(item)
        return pd.get_dummies(df.drop(drop_items, axis=1))

    @classmethod
    def get_missing_list(cls, df: pd.DataFrame):

        null_list = list([])
        for index, isnull in pd.isnull(df).any(axis=0).iteritems():
            if isnull:
                null_list.append(index)

        return null_list

    @classmethod
    def get_df_filled(cls, df: pd.DataFrame, missing_list: list, method='median'):

        for missing in missing_list:
            if method == 'median':
                df[missing] = df[missing].fillna(df[missing].median())
            elif method == 'mean':
                df[missing] = df[missing].fillna(df[missing].mean())
        return df
