from texttable import Texttable


class ShowTexttable:

    def show_texttable(df, cols_align, cols_valign, cols_dtype):

        table = Texttable()

        if len(cols_align) > 0:
            table.set_cols_align(cols_align)
        if len(cols_valign) > 0:
            table.set_cols_valign(cols_valign)
        if len(cols_dtype) > 0:
            table.set_cols_dtype(cols_dtype)

        if type(df) is pd.DataFrame:
            df = df.values

        table.add_rows(df)
        return table.draw()


    import pandas as pd

    df = pd.read_csv('train.csv')
    cols_align = []
    cols_valign = []
    cols_dtype = []
    print(show_texttable(df.describe(include='all'), cols_align, cols_valign, cols_dtype))
