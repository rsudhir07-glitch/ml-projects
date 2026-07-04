import pandas as pd


def preprocess_data() : 
    data = pd.read_csv('data/marketing_campaign.csv' , sep='\t')
    data = data.drop(columns = ['ID' , 'Year_Birth'])
    data['Income'] = data['Income'].fillna(data['Income'].mean())
   


    object_columns = data.select_dtypes(include = 'object').columns
    
    # now we will one hot encode the categorical colummns
    data = pd.get_dummies(data , columns = object_columns)

    return data

