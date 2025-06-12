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