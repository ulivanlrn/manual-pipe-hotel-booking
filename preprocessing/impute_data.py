def fill_in_constant(column, value):
    return column.fillna(value)

def random_impute(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isna()].index
    df.loc[df[feature].isna(), feature] = random_sample

def run_imputation(data, config, current_features):
    if 'children' in current_features:
        data['children'] = fill_in_constant(data['children'],
                                            config["preprocessing"]["children_impute_value"])
    # NOTE: directly depends on the sample
    for feature in ['country', 'agent']:
        if feature in current_features:
            random_impute(data, feature)

    return data