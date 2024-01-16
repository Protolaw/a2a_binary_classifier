import pandas as pd
from DataPreparation import DataPreparation

from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/balanced_dataframe.csv', index_col=None)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=y
                                                    )
y_test.value_counts()

data_preparation = DataPreparation(X_train, X_test, y_train, y_test)

# # get scaled clean data
# # use StandartScaler
X_train, X_test, y_train, y_test = data_preparation.clean_dataset()

print(X_train.columns)
print(X_train.shape)