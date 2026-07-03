from sklearn.model_selection import train_test_split


def trained_model(data):
    X = data.drop('Response')
    y = data['Response']
    X_train , X_test , y_train , y_test = train_test_split(
       X , y,
       test_size= 0.3,
       random_state = 42,
       stratify = y

    )

    # now we will use stratified kflod here






    return 