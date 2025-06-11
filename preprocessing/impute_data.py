def fill_in_constant(column, value):
    return column.fillna(value)

def random_impute(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isna()].index
    df.loc[df[feature].isna(), feature] = random_sample