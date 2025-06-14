from utils import set1_in_set2

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

def run_feature_engineering(data, config, current_features):
    feature_flags = config["features"]["flags"]
    # a variable that collects all old features which contributed to new ones \
    # and are not needed afterward
    drop_after_fe = set()

    if feature_flags["total_nights"] & \
            set1_in_set2({'stays_in_weekend_nights', 'stays_in_week_nights'}, current_features):
        data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
        drop_after_fe = drop_after_fe.union({'stays_in_weekend_nights', 'stays_in_week_nights'})

    if feature_flags["stays_format"] & \
            set1_in_set2({'stays_in_weekend_nights', 'stays_in_week_nights'}, current_features):
        data['stays_format'] = data.apply(stays_func, axis=1)

    if feature_flags["total_guests"] & \
            set1_in_set2({'adults', 'children', 'babies'}, current_features):
        data['total_guests'] = data['adults'] + data['children'] + data['babies']
        drop_after_fe = drop_after_fe.union({'adults', 'children', 'babies'})

    if feature_flags["room_assigned_equal_reserved"] & \
            set1_in_set2({'assigned_room_type', 'reserved_room_type'}, current_features):
        data['room_assigned_equal_reserved'] = data.apply(room_type, axis=1)
        drop_after_fe = drop_after_fe.union({'assigned_room_type', 'reserved_room_type'})

    if feature_flags["map_deposit_type"]:
        data['deposit_type'] = data.apply(deposit_type, axis=1)

    data.drop(drop_after_fe, axis=1, inplace=True)

    return data