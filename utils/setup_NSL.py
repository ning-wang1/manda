import pandas as pd
import numpy as np
import itertools
import sys
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# from keras.models import Sequential
# from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils.classifier import evaluate_sub
import random

FEATURE_NUM = 40  # the number of selected features
VALIDATION_SIZE = 5000
# file paths of training and testing data
# train_file_path = 'NSL_KDD/KDDTrain+_20Percent.txt'
# test_file_path = 'NSL_KDD/KDDTest-21.txt'

sys.path.append('../')
train_file_path = 'data/NSL_KDD/KDDTrain+.txt'
test_file_path = 'data/NSL_KDD/KDDTest+.txt'
SCALER = 'std_scale'

# attributes/features of the data
datacols = ["duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"]

datacols_no_outbound = ["duration", "protocol_type", "service", "flag", "src_bytes",
                        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                        "num_file_creations", "num_shells", "num_access_files",
                        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]


class MinMax:
    def __init__(self, max_v, min_v):
        self.min_v = min_v
        self.max_v = max_v

    def transform(self, x):
        y = (x - self.min_v) / (self.max_v - self.min_v)
        return y

    def inverse_transform(self, y):
        x = y * (self.max_v - self.min_v) + self.min_v
        return x


def preprocessing(scaler_name=SCALER):
    """ Loading data
    """
    #  Load NSL_KDD train dataset
    df_train = pd.read_csv(train_file_path, sep=",", names=datacols)  # load data
    df_train = df_train.iloc[:, :-1]  # removes an unwanted extra field

    # Load NSL_KDD test dataset
    df_test = pd.read_csv(test_file_path, sep=",", names=datacols)
    df_test = df_test.iloc[:, :-1]

    # train set dimension
    print('Train set dimension: {} rows, {} columns'.format(df_train.shape[0], df_train.shape[1]))
    # test set dimension
    print('Test set dimension: {} rows, {} columns'.format(df_test.shape[0], df_test.shape[1]))

    datacols_range_continous = {"duration": 58329.0, "src_bytes": 1379963888.0, "dst_bytes": 1309937401.0,
                                "wrong_fragment": 3.0, "urgent": 14.0, "hot": 101.0, "num_failed_logins": 5.0,
                                "num_compromised": 7479.0, "num_root": 7468.0, "num_file_creations": 100.0,
                                "num_shells": 5.0,
                                "num_access_files": 9.0, "num_outbound_cmds": 0.0, "count": 511.0, "srv_count": 511.0,
                                "serror_rate": 1.0, "srv_serror_rate": 1.0, "rerror_rate": 1.0, "srv_rerror_rate": 1.0,
                                "same_srv_rate": 1.0, "diff_srv_rate": 1.0, "srv_diff_host_rate": 1.0,
                                "dst_host_count": 255.0,
                                "dst_host_srv_count": 255.0, "dst_host_same_srv_rate": 1.0,
                                "dst_host_diff_srv_rate": 1.0,
                                "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 1.0,
                                "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
                                "dst_host_rerror_rate": 1.0,
                                "dst_host_srv_rerror_rate": 1.0, "land": 1.0, "logged_in": 1.0, "root_shell": 1.0,
                                "su_attempted": 1.0, "is_host_login": 1.0, "is_guest_login": 1.0}

    datacols_range_discrere = {"land": 1, "logged_in": 1, "root_shell": 1, "su_attempted": 1, "is_host_login": 1,
                               "is_guest_login": 1}
    #  data preprocessing
    mapping = {'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'saint': 'Probe',
               'mscan': 'Probe',
               'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS', 'neptune': 'DoS', 'smurf': 'DoS',
               'mailbomb': 'DoS',
               'udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
               'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R', 'xterm': 'U2R',
               'ps': 'U2R',
               'sqlattack': 'U2R', 'httptunnel': 'U2R',
               'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
               'imap': 'R2L',
               'spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L',
               'snmpgetattack': 'R2L',
               'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
               'normal': 'Normal'
               }

    # Apply attack class mappings to the dataset
    df_train['attack_class'] = df_train['attack'].apply(lambda v: mapping[v])
    df_test['attack_class'] = df_test['attack'].apply(lambda v: mapping[v])

    # Drop attack field from both train and test data
    df_train.drop(['attack'], axis=1, inplace=True)
    df_test.drop(['attack'], axis=1, inplace=True)

    # 'num_outbound_cmds' field has all 0 values. Hence, it will be removed from both train and test dataset
    df_train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    df_test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    # Attack Class Distribution
    attack_class_freq_train = df_train[['attack_class']].apply(lambda x: x.value_counts())
    attack_class_freq_test = df_test[['attack_class']].apply(lambda x: x.value_counts())
    attack_class_freq_train['frequency_percent_train'] = round(
        (100 * attack_class_freq_train / attack_class_freq_train.sum()), 2)
    attack_class_freq_test['frequency_percent_test'] = round(
        (100 * attack_class_freq_test / attack_class_freq_test.sum()), 2)

    attack_class_dist = pd.concat([attack_class_freq_train, attack_class_freq_test], sort=False, axis=1)
    # print(attack_class_dist)

    # Attack class bar plot
    # plot = attack_class_dist[['frequency_percent_train', 'frequency_percent_test']].plot(kind="bar")
    # plot.set_title("Attack Class Distribution", fontsize=20)
    # plot.grid(color='lightgray', alpha=0.5)

    if scaler_name is 'std_scale':
        # Scaling Numerical Attributes
        scaler = StandardScaler()
        # extract numerical attributes and scale it to have zero mean and unit variance
        cols = df_train.select_dtypes(include=['float64', 'int64']).columns
        print(len(cols))
        scaler.fit(df_train[cols])
        train_numerical = scaler.transform(df_train[cols])
        test_numerical = scaler.transform(df_test[cols])

        # calculate the min value and max value of different features
        max_v = []
        max_dic = {}
        [max_v.append(datacols_range_continous[col]) for col in cols]
        max_v_df = pd.DataFrame(np.array(max_v).reshape(1, len(max_v)), columns=cols)
        max_v = scaler.transform(max_v_df)

        for i in range(max_v.shape[1]):
            max_dic[cols[i]] = max_v[0, i]

        min_v = []
        min_dic = {}
        [min_v.append(0) for col in cols]
        min_v_df = pd.DataFrame(np.array(min_v).reshape(1, len(min_v)), columns=cols)
        min_v = scaler.transform(min_v_df)
        # check = scaler.inverse_transform(min_v)

        for i in range(min_v.shape[1]):
            min_dic[cols[i]] = min_v[0, i]
    else:
        # extract numerical attributes and scale it to have zero mean and unit variance
        cols = df_train.select_dtypes(include=['float64', 'int64']).columns
        max_v = []
        min_v = []
        [max_v.append(datacols_range_continous[col]) for col in cols]
        [min_v.append(0) for col in cols]
        scaler = MinMax(max_v=np.array(max_v), min_v=np.array(min_v))
        train_numerical = scaler.transform(df_train[cols].values)
        test_numerical = scaler.transform(df_test[cols].values)
        min_dic = {}
        max_dic = {}
        for i in range(len(min_v)):
            min_dic[cols[i]] = 0
            max_dic[cols[i]] = 1.0

    # turn the result back to a data frame
    train_numerical_df = pd.DataFrame(train_numerical, columns=cols)
    test_numerical_df = pd.DataFrame(test_numerical, columns=cols)

    # Encoding of categorical Attributes
    encoder = LabelEncoder()
    # extract categorical attributes from both training and test sets
    cat_train = df_train.select_dtypes(include=['object']).copy()
    cat_test = df_test.select_dtypes(include=['object']).copy()
    # encode the categorical attributes
    train_categorical = cat_train.apply(encoder.fit_transform)
    test_categorical = cat_test.apply(encoder.fit_transform)

    # data sampling
    # define columns and extract encoded train set for sampling
    x_train_categorical = train_categorical.drop(['attack_class'], axis=1)
    class_col = pd.concat([train_numerical_df, x_train_categorical], axis=1).columns
    x_train = np.concatenate((train_numerical, x_train_categorical.values), axis=1)
    x = x_train
    y_train = train_categorical[['attack_class']].copy()
    c, r = y_train.values.shape
    y = y_train.values.reshape(c, )

    # apply the random over-sampling
    # ros = RandomOverSampler(random_state=42)
    # x, y = ros.fit_sample(x, y)

    # create test data frame
    test = pd.concat([test_numerical_df, test_categorical], axis=1)
    test['attack_class'] = test['attack_class'].astype(np.float64)
    test['protocol_type'] = test['protocol_type'].astype(np.float64)
    test['flag'] = test['flag'].astype(np.float64)
    test['service'] = test['service'].astype(np.float64)

    print('Original dataset shape {}'.format(Counter(df_train)))
    print('Resampled dataset shape {}'.format(x.shape))

    return x, y, class_col, test, max_dic, min_dic, scaler


def oneHot(train_data, test_data, features, col_names=('protocol_type', 'service', 'flag')):
    # translate the features to one hot
    enc = OneHotEncoder()
    train_data = train_data[features]
    test_data = test_data[features]
    category_max = [3, 70, 11]
    x_train_1hot = []
    x_test_1hot = []
    cat_num_dict ={}
    for col in col_names:
        print(col)
        if col in train_data.columns:  # split the columns to 2 set: one for numerical, another is categorical
            train_data_num = train_data.drop([col], axis=1)
            train_data_cat = train_data[[col]].copy()
            test_data_num = test_data.drop([col], axis=1)
            test_data_cat = test_data[[col]].copy()

            # Fit train data
            enc.fit(train_data_cat.append(test_data_cat))
            x_train_1hot.append(enc.transform(train_data_cat).toarray())
            x_test_1hot.append(enc.transform(test_data_cat).toarray())

            train_data = train_data_num
            test_data = test_data_num
        cat_num_dict[col] = x_train_1hot[-1].shape[1]

    x_train = train_data_num.values
    x_test = test_data_num.values

    for train_1hot, test_1hot in zip(x_train_1hot, x_test_1hot):
        x_train = np.concatenate((x_train, train_1hot), axis=1)
        x_test = np.concatenate((x_test, test_1hot), axis=1)
    return x_train, x_test, cat_num_dict


def cat_to_num(train_data, test_data, features, col_names=('protocol_type', 'service', 'flag')):
    # translate the features to one hot
    enc = OneHotEncoder()
    train_data = train_data[features]
    test_data = test_data[features]
    category_max = [3, 70, 11]
    x_train_1hot = []
    x_test_1hot = []

    for col in col_names:
        print(col)
        if col in train_data.columns:  # split the columns to 2 set: one for numerical, another is categorical
            train_data_num = train_data.drop([col], axis=1)
            train_data_cat = train_data[[col]].copy()
            test_data_num = test_data.drop([col], axis=1)
            test_data_cat = test_data[[col]].copy()

            # Fit train data
            enc.fit(train_data_cat.append(test_data_cat))
            x_train_1hot.append(enc.transform(train_data_cat).toarray())
            x_test_1hot.append(enc.transform(test_data_cat).toarray())

            train_data = train_data_num
            test_data = test_data_num

    x_train = train_data_num.values
    x_test = test_data_num.values

    for train_1hot, test_1hot in zip(x_train_1hot, x_test_1hot):
        x_train = np.concatenate((x_train, train_1hot), axis=1)
        x_test = np.concatenate((x_test, test_1hot), axis=1)
    return x_train, x_test


def data_partition(x, y, class_col, test, features, attack_class):
    """data partition to 
    """
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PARTITION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    attack_name = attack_class[0][0]
    new_col = list(class_col)
    new_col.append('attack_class')

    # add a dimension to target
    new_y = y[:, np.newaxis]

    # create a data frame from sampled data
    data_arr = np.concatenate((x, new_y), axis=1)
    data_df = pd.DataFrame(data_arr, columns=new_col)

    # x_train, x_test = oneHot(data_df, test, features)

    # create two-target classes (normal class and an attack class)
    class_dict = defaultdict(list)
    normal_class = [('Normal', 1.0)]

    class_dict = create_class_dict(class_dict, data_df, test, normal_class, attack_class)
    train_data = class_dict['Normal_' + attack_name][0]
    test_data = class_dict['Normal_' + attack_name][1]
    grpclass = 'Normal_' + attack_name

    # transform the selected features to one-hot
    x_train, x_test, cat_num_dict = oneHot(train_data, test_data, features)

    y_train = train_data[['attack_class']].copy()
    c, r = y_train.values.shape
    y_train = y_train.values.reshape(c, )
    y_test = test_data[['attack_class']].copy()
    c, r = y_test.values.shape
    y_test = y_test.values.reshape(c, )
    # transform the labels to one-hot
    y_train = [attack_class[0][1], 1] == y_train[:, None].astype(np.float32)
    y_test = [attack_class[0][1], 1] == y_test[:, None].astype(np.float32)
    print('x_train', x_train[0], x_train[0].shape)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DATA PARTITION FINISHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    return x_train, x_test, y_train, y_test, cat_num_dict


def create_class_dict(class_dict, data_df, test_df, normal_class, attack_class):
    """ This function subdivides train and test dataset into two-class attack labels
    return the loc of target attack and normal samples
    """
    j = normal_class[0][0]  # name of normal class
    k = normal_class[0][1]  # numerical representer of normal class
    i = attack_class[0][0]  # name of abnormal class(DOS, Probe, R2L, U2R)
    v = attack_class[0][1]  # numerical represent of abnormal classes 
    # [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]
    train_set = data_df.loc[(data_df['attack_class'] == k) | (data_df['attack_class'] == v)]
    # transform the selected features to one-hot

    class_dict[j + '_' + i].append(train_set)
    # test labels
    test_set = test_df.loc[(test_df['attack_class'] == k) | (test_df['attack_class'] == v)]
    class_dict[j + '_' + i].append(test_set)

    return class_dict


def create_class_dict_balance(class_dict, data_df, test_df, normal_class, attack_class):
    """ This function subdivides train and test dataset into two-class attack labels
    return the loc of target attack and normal samples
    """
    j = normal_class[0][0]  # name of normal class
    k = normal_class[0][1]  # numerical representer of normal class
    i = attack_class[0][0]  # name of abnormal class(DOS, Probe, R2L, U2R)
    v = attack_class[0][1]  # numerical represent of abnormal classes
    # [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]
    train_set_normal = data_df.loc[(data_df['attack_class'] == k)]
    train_set_anormal = data_df.loc[(data_df['attack_class'] == v)]
    df1 = train_set_normal.sample(frac=0.25)
    train_set = pd.concat([train_set_anormal, df1])
    class_dict[j + '_' + i].append(train_set)
    # test labels
    test_set = test_df.loc[(test_df['attack_class'] == k) | (test_df['attack_class'] == v)]
    class_dict[j + '_' + i].append(test_set)

    return class_dict


def feature_selection(x, y, x_col_name, FEATURE_NUM):
    """ feature selections
    """
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FEATURE SELECTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(x.shape)
    print(y.shape)
    rfc = RandomForestClassifier()
    # fit random forest classifier on the training set
    y = y.reshape(-1, 1)  # reshape the labels
    rfc.fit(x, y)
    # extract important features
    score = np.round(rfc.feature_importances_, 3)
    significance = pd.DataFrame({'feature': x_col_name, 'importance': score})
    significance = significance.sort_values('importance', ascending=False).set_index('feature')
    # plot significance
    plt.rcParams['figure.figsize'] = (11, 4)
    significance.plot.bar()

    # create the RFE model and select 10 attributes
    rfe = RFE(rfc, FEATURE_NUM)
    rfe = rfe.fit(x, y)

    # summarize the selection of the attributes
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), x_col_name)]
    selected_features = [v for i, v in feature_map if i == True]

    return selected_features


class NSL_KDD:
    def __init__(self, attack_class, num_selected_fea=0):

        x, y, x_col_name, test, max_dic, min_dic, scaler = preprocessing()

        if not num_selected_fea == 0:
            selected_features = feature_selection(x, y, x_col_name, FEATURE_NUM)
        else:
            selected_features = datacols_no_outbound

        print('selected_feature>>>>>>>>\n.', selected_features)
        x_train, x_test, y_train, y_test, cat_num_dict = \
            data_partition(x, y, x_col_name, test, selected_features, attack_class)

        max_v = []
        min_v = []
        categorical_features = ['protocol_type', 'service', 'flag']  # 3, 70, 11

        numerical_features = []
        for col in datacols:
            if col in selected_features and not col in categorical_features:
                numerical_features.append(col)

        [max_v.append(max_dic[col]) for col in numerical_features]
        [min_v.append(min_dic[col]) for col in numerical_features]
        max_v.extend(np.ones(x_train.shape[1] - len(numerical_features)))
        min_v.extend(np.zeros(x_train.shape[1] - len(numerical_features)))
        # if 'protocol_type' in selected_features:
        #     max_v.extend(np.ones(3))
        #     min_v.extend(np.zeros(3))
        # if 'service' in selected_features:
        #     max_v.extend(np.ones(70))
        #     min_v.extend(np.zeros(70))
        # if 'flag' in selected_features:
        #     max_v.extend(np.ones(11))
        #     min_v.extend(np.zeros(11))

        self.test_data = x_test
        self.test_labels = y_test
        self.cat_num_dict = cat_num_dict
        self.validation_data = x_train[:VALIDATION_SIZE, ]
        self.validation_labels = y_train[:VALIDATION_SIZE]
        self.train_data = x_train[VALIDATION_SIZE:, ]
        self.train_labels = y_train[VALIDATION_SIZE:]
        self.max_v = np.array(max_v)  # the maximum value of each numerical feature
        self.min_v = np.array(min_v)  # the minimum value of each numerical feature
        self.scaler = scaler
        self.FEATURE_NUM_FINAL = x_train.shape[1]

    def data_rerange(self, data):
        scaler = MinMaxScaler()
        # extract numerical attributes and scale it to have zero mean and unit variance
        data = scaler.fit_transform(data) - 0.5
        return data

    def get_feature_mean(self):
        y = np.argmax(self.train_labels, axis=1)
        pos = np.where(y==0)[0]
        x = self.train_data[pos, :37]
        x_inverse = self.scaler.inverse_transform(x)
        feature_mean_val = np.mean(x_inverse, axis=0)
        print(feature_mean_val.shape)
        print(feature_mean_val)
        return feature_mean_val


class NSLModel:
    def __init__(self, restore, feature_num, session=None):
        self.num_features = feature_num
        self.num_labels = 2

        model = Sequential()
        model.add(Dense(50, input_dim=feature_num, activation='relu'))
        model.add(Dense(2))
        model.load_weights(restore)
        self.model = model

    def predict(self, data):
        return self.model(data)

    def evaluate_only(self, x, y):
        outputs = self.model(x).eval()
        y_pred = np.argmax(outputs, axis=1)
        evaluate_sub('nn model', y, y_pred)

    def evaluate(self, x, y):
        predicted = self.model(x).eval()
        acc = np.count_nonzero(predicted.argmax(1) == y.argmax(1)) / y.shape[0]
        return acc

    def save_weights(self, file_path):
        self.model.save_weights(file_path)

    def get_weights(self):
        self.model.get_weights()