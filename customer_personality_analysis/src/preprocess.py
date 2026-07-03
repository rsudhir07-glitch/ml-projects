import pandas as pd


def preprocess_data() : 
    data = pd.read_csv          ('customer_personality_analysis/data/marketing_campaign.csv' , sep='\t')

    data['Income'] = data['Income'].fillna(data['Income'].mean)
    # print(data.isnull().sum())


    object_type = data.select_dtypes(include = 'object').columns

    # now we will one hot encode the categorical colummns

    object_type = pd.get_dummies(data , columns = object_type)

    return data

#print(data.isnull().sum())
