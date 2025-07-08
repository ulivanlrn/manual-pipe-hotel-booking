from utils.utils import set1_in_set2

def stays_func(row):
    feature1 = 'stays_in_weekend_nights'
    feature2 = 'stays_in_week_nights'

    if (row[feature1] > 0) & (row[feature2] == 0):
        return 'just_weekend'
    if (row[feature1] == 0) & (row[feature2] > 0):
        return 'just_week'
    if (row[feature1] > 0) & (row[feature2] > 0):
        return 'both_weekend_and_week'
    else:
        return 'undefined'

def room_type(row):
    if row['assigned_room_type'] == row['reserved_room_type']:
        return 1
    else:
        return 0

def deposit_type(row):
    if row['deposit_type'] == 'Non Refund':
        return 1
    else:
        return 0

def run_feature_engineering(data, config, current_features: set):
    """
    Run feature engineering on the given data.
    :param data: Dataframe.
    :param config: Configuration file.
    :param current_features: The features which are currently in the data.
    :return: Dataframe enriched with the new features.
    """
    df = data.copy()

    # A dictionary specifying which features should be created
    feature_flags = config["feature_flags"]

    # a variable that collects all old features which contributed to new ones \
    # and are not needed afterward
    feats_to_drop = set()

    if feature_flags["total_nights"] & \
            set1_in_set2({'stays_in_weekend_nights', 'stays_in_week_nights'}, current_features):
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        feats_to_drop = feats_to_drop.union({'stays_in_weekend_nights', 'stays_in_week_nights'})

    if feature_flags["stays_format"] & \
            set1_in_set2({'stays_in_weekend_nights', 'stays_in_week_nights'}, current_features):
        df['stays_format'] = df.apply(stays_func, axis=1)

    if feature_flags["total_guests"] & \
            set1_in_set2({'adults', 'children', 'babies'}, current_features):
        df['total_guests'] = df['adults'] + df['children'] + df['babies']
        feats_to_drop = feats_to_drop.union({'adults', 'children', 'babies'})

    if feature_flags["room_assigned_equal_reserved"] & \
            set1_in_set2({'assigned_room_type', 'reserved_room_type'}, current_features):
        df['room_assigned_equal_reserved'] = df.apply(room_type, axis=1)
        feats_to_drop = feats_to_drop.union({'assigned_room_type', 'reserved_room_type'})

    if feature_flags["map_deposit_type"]:
        df['deposit_type'] = df.apply(deposit_type, axis=1)

    df.drop(feats_to_drop, axis=1, inplace=True)
    return df