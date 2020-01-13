import sys
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class SalaryRegression(object):
    def __init__(self, train_features_file, train_salaries_file, test_features_file):
        self.train_features_file = train_features_file
        self.train_salaries_file = train_salaries_file
        self.test_features_file = test_features_file
        self.train_features = None
        self.train_salaries = None
        self.test_features = None
        self.ordinal_encoder = OrdinalEncoder()
        self.label_encoder = preprocessing.LabelEncoder()
        self.onehot_encoder = OneHotEncoder(dtype=np.int, sparse=True)

    def read_data_files(self):
        self.train_features = pd.read_csv(self.train_features_file)
        self.train_salaries = pd.read_csv(self.train_salaries_file)
        self.test_features = pd.read_csv(self.test_features_file)

    def analyze_feature_correlation(self):
        data_train = pd.merge(self.train_features, self.train_salaries, on='jobId', how='outer')

        le = preprocessing.LabelEncoder()
        data_train = data_train.apply(le.fit_transform)

        columns = list(data_train.columns.values)
        columns.remove("jobId")

        correlation_map = np.corrcoef(data_train[columns].values.T)
        sns.set(font_scale=1.0)
        heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f',
                              yticklabels=columns, xticklabels=columns)
        plt.show()

    def get_train_data(self):
        data_train = pd.merge(self.train_features, self.train_salaries, on='jobId', how='outer')
        data_train = data_train.replace("NONE", np.nan)

        row, col = data_train.shape

        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer = imputer.fit(data_train.loc[:int(row / 4), 'degree':'major'])
        data_train.loc[:, 'degree':'major'] = imputer.transform(data_train.loc[:, 'degree':'major'])

        jobType_unique = sorted(data_train['jobType'].unique())
        major_unique = sorted(data_train['major'].unique())
        industry_unique = sorted(data_train['industry'].unique())

        nominal_cols = jobType_unique + major_unique + industry_unique

        cat = pd.Categorical(data_train.degree,
                             categories=['HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'DOCTORAL'],
                             ordered=True)

        labels, unique = pd.factorize(cat, sort=True)
        data_train.degree = labels

        data_train.degree = self.ordinal_encoder.fit_transform(data_train.degree.values.reshape(-1, 1))
        nominals = pd.DataFrame(
            self.onehot_encoder.fit_transform(data_train[['jobType', 'major', 'industry']]).toarray(),
            columns=[nominal_cols])

        nominals['yearsExperience'] = data_train['yearsExperience']
        nominals['milesFromMetropolis'] = data_train['milesFromMetropolis']
        nominals['milesFromMetropolis'] = nominals['milesFromMetropolis'].apply(lambda x: -x)

        disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
        disc.fit(nominals.loc[:, 'yearsExperience'])
        nominals.loc[:, 'yearsExperience'] = disc.transform(nominals.loc[:, 'yearsExperience'])

        disc = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        disc.fit(nominals.loc[:, 'milesFromMetropolis'])
        nominals.loc[:, 'milesFromMetropolis'] = disc.transform(nominals.loc[:, 'milesFromMetropolis'])

        y_train = data_train['salary']
        x_train = nominals
        x_train = csr_matrix(x_train.values)

        return x_train, y_train

    def get_test_data(self):
        data_test = self.test_features
        data_test = data_test.replace("NONE", np.nan)

        row, col = data_test.shape

        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer = imputer.fit(data_test.loc[:int(row / 4), 'degree':'major'])
        data_test.loc[:, 'degree':'major'] = imputer.transform(data_test.loc[:, 'degree':'major'])

        jobType_unique = sorted(data_test['jobType'].unique())
        major_unique = sorted(data_test['major'].unique())
        industry_unique = sorted(data_test['industry'].unique())

        nominal_cols = jobType_unique + major_unique + industry_unique

        cat = pd.Categorical(data_test.degree,
                             categories=['HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'DOCTORAL'],
                             ordered=True)

        labels, unique = pd.factorize(cat, sort=True)
        data_test.degree = labels

        data_test.degree = self.ordinal_encoder.fit_transform(data_test.degree.values.reshape(-1, 1))
        nominals = pd.DataFrame(
            self.onehot_encoder.fit_transform(data_test[['jobType', 'major', 'industry']]).toarray(),
            columns=[nominal_cols])

        nominals['yearsExperience'] = data_test['yearsExperience']
        nominals['milesFromMetropolis'] = data_test['milesFromMetropolis']
        nominals['milesFromMetropolis'] = nominals['milesFromMetropolis'].apply(lambda x: -x)

        disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
        disc.fit(nominals.loc[:, 'yearsExperience'])
        nominals.loc[:, 'yearsExperience'] = disc.transform(nominals.loc[:, 'yearsExperience'])

        disc = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        disc.fit(nominals.loc[:, 'milesFromMetropolis'])
        nominals.loc[:, 'milesFromMetropolis'] = disc.transform(nominals.loc[:, 'milesFromMetropolis'])

        x_test = nominals
        x_test = csr_matrix(x_test.values)

        return x_test

    def check_algorithm_performance(self):
        X, Y = self.get_train_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        pipelines = list()
        pipelines.append(('LinearRegression', Pipeline([('LR', LinearRegression())])))
        pipelines.append(('PolynomialLinearRegression', Pipeline([('PLR', PolynomialFeatures(degree=2)),
                                                                   ('linear', linear_model.LinearRegression(
                                                                       fit_intercept=False))])))
        pipelines.append(('RandomForestRegressor', Pipeline([('RF',
                                                              RandomForestRegressor(n_estimators=10,
                                                                                    n_jobs=6,
                                                                                    max_depth=20))])))
        results = []
        names = []
        print("Cross-validation accuracies of various models")
        for name, model in pipelines:
            kfold = KFold(n_splits=5, random_state=21)
            cv_results = np.sqrt(
                -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error'))
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f" % (name, cv_results.mean())
            print(msg)

    def predict(self):
        X, Y = self.get_train_data()
        x_test = self.get_test_data()
        model = RandomForestRegressor(n_estimators=10, n_jobs=6, max_depth=20)
        model.fit(X, Y)
        y_pred = model.predict(x_test)

        output_file = open("./test_salaries.csv", "w+")
        for pred in y_pred:
            output_file.write("{}\n".format(pred))
        output_file.close()


def main():
    pd.set_option('display.max_columns', None)
    train_features_file = sys.argv[1]
    train_salaries_file = sys.argv[2]
    test_features_file = sys.argv[3]

    sg = SalaryRegression(train_features_file, train_salaries_file, test_features_file)
    sg.read_data_files()
    sg.analyze_feature_correlation()
    sg.check_algorithm_performance()
    sg.predict()


if __name__ == '__main__':
    main()
