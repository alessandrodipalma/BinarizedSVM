import pandas as pd

def load(dataset):
    return DATASETS_DICT[dataset]()

def load_all():
    all_datasets = []
    for dataset, load_fun in DATASETS_DICT.items():
        X, y, features_names = load_fun()
        all_datasets.append(
            { 'name': dataset,
              'X': X,
              'y': y,
              'features_names': features_names
            }
        )

    return all_datasets
def load_sonar():
    df = pd.read_csv('datasets/sonar.all-data', header=None)
    # trasformo le etichette da [1,2] a [-1,1]
    y = df[60].replace({'M': 1, 'R': -1}).values
    # prendo solo le prime 6 colonne, che corrispondono a quelle delle feature
    X = df.values[:, :60]

    features_names = df.columns.values
    return X, y, features_names


def load_bands():
    df = pd.read_csv('datasets/bands.data', header=None)
    df = df[~df.isin(['?']).any(axis=1)]
    df.replace({'YES': 1, 'NO': 0}, inplace=True)
    df.replace({'CANAdiAN': 'CANADIAN'}, inplace=True)
    df = pd.get_dummies(df, columns=[7, 9, 10, 12, 14, 15, 17, 18]).drop(columns=[0, 1, 2, 5, 8]).reset_index()
    y = df[39].replace({'band': 1, 'noband': -1}).values
    X = df.drop(columns=[39]).values.astype(float)
    features_names = df.drop(columns=[39]).columns.values
    return X, y, features_names


def load_credit():
    return X, y, features_names


def load_ionosphere():
    df = pd.read_csv("datasets/ionosphere.data", header=None)
    X = df.values[:, :-1]
    y = df[34].apply(lambda x: 1 if x == 'g' else -1).values
    features_names = [str(i) for i in range(X.shape[0])]
    return X, y, features_names


def load_wdbc():
    df = pd.read_csv("datasets/wdbc.csv")
    X = df.values[:, 2:-1]
    y = df["diagnosis"].apply(lambda x: 1 if x == 'M' else -1).values
    features_names = df.columns.values[2:-1]
    return X, y, features_names


def load_cleveland():
    df = pd.read_csv('datasets/processed.cleveland.data', header=None)
    df = df[(df[12] != '?') & (df[11] != '?')]
    X = df.values[:, :13].astype(float)
    y = df[13].values
    features_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                      'ca', 'thal']

    return X, y, features_names


def load_housing():
    df = pd.read_csv("datasets/HousingData.csv")
    df = df[~df.isnull().any(axis=1)]
    X = df.values[:, :-1]
    y = df["MEDV"].apply(lambda x: 1 if x > 21 else -1).values
    features_names = df.columns.values[:-1]
    return X, y, features_names

def load_pima():
    df = pd.read_csv("datasets/diabetes.csv")
    X = df.values[:, :-1]
    y = df["Outcome"].apply(lambda x: 1 if x == 1 else -1).values
    features_names = df.columns.values[:-1]
    return X, y, features_names


def load_bupa():
    df = pd.read_csv('datasets/bupa.data', header=None)
    # trasformo le etichette da [1,2] a [-1,1]
    y = df[6].replace({2:1,1:-1}).values
    # prendo solo le prime 6 colonne, che corrispondono a quelle delle feature
    X = df.values[:, :6]

    features_names = [str(i) for i in range(X.shape[0])]

    return X, y, features_names


DATASETS_DICT = {
    'sonar': load_sonar,
    'bands': load_bands,
    #'credit': load_credit,
    'ionosphere': load_ionosphere,
    'wdbc': load_wdbc,
    #'cleveland': load_cleveland,
    'housing': load_housing,
    'pima': load_pima,
    'bupa': load_bupa
}
