import pandas as pd

from sklearn import utils
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import preprocessing


class DataSet:

    def __init__(self, actives, decoys, *, threshold=.7, test_size=.3, random_state=42):

        self.actives = actives
        self.decoys = decoys
        self.test_size = test_size
        self.threshold = threshold
        self.random_state = random_state

    def _create_data_frame(self):
        number_actives = self.actives.shape[0]
        self.decoys = utils.resample(self.decoys, n_samples=number_actives)
        self.actives['target'] = 1
        self.decoys['target'] = 0

        dataframe = pd.concat([self.actives, self.decoys])
        dataframe.set_index('SMILES', inplace=True)

        return dataframe

    def _split(self, dataframe):
        X, y = dataframe.drop(columns='target'), dataframe['target']
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            shuffle=True,
                                                            stratify=y,
                                                            random_state=self.random_state
                                                            )
        return X_train, X_test, y_train, y_test

    def create_train_test_data(self):
        df = self._create_data_frame()

        return self._split(df)


class DataPreparation:

    def __init__(self, X_train, X_test, y_train, y_test, *, threshold=.7):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.threshold = threshold

    def remove_collinear_features_train(self):

    # '''
    #     Objective:
    #         Remove collinear features in a dataframe with a correlation coefficient
    #         greater than the threshold. Removing collinear features can help a model 
    #         to generalize and improves the interpretability of the model.

    #     Inputs: 
    #         x: features dataframe
    #         threshold: features with correlations greater than this value are removed

    #     Output: 
    #         dataframe that contains only the non-highly-collinear features
    #     '''

        # Calculate the correlation matrix
        corr_matrix = self.X_train.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val > self.threshold:
                    # Print the correlated features and the correlation value
                    #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)

        return drops

    def remove_collinear_features(self, drops):
        self.X_train = self.X_train.drop(columns=drops)
        self.X_test = self.X_test.drop(columns=drops)

        return self.X_train, self.X_test

    def remove_constant_features(self):

        var_tresh = feature_selection.VarianceThreshold(threshold=0.7)

        self.X_train = var_tresh.fit_transform(self.X_train)
        self.X_test = var_tresh.transform(self.X_test)

        return self.X_train, self.X_test

    def scaling(self):

        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(self.X_train)

        self.X_train = pd.DataFrame(scaler.transform(self.X_train))
        self.X_test = pd.DataFrame(scaler.transform(self.X_test))
        return self.X_train, self.X_test

    def clean_dataset(self):

        drops = self.remove_collinear_features_train()
        
        self.X_train, self.X_test = self.remove_collinear_features(drops)

        self.X_train, self.X_test = self.scaling()

        self.X_train, self.X_test = self.remove_constant_features()

        return pd.DataFrame(self.X_train), pd.DataFrame(self.X_test), self.y_train, self.y_test