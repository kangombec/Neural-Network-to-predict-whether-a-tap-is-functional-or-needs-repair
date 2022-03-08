import pandas as pd
import pandas as np
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network._multilayer_perceptron import MLPClassifier

# load the dataset function
warnings.filterwarnings('ignore')


def load_dataset(filename):
    df = pd.read_csv(filename)
    return df


df = load_dataset('task1_train.csv')
print(df.sample(10))
# check the class distribution of the target variable
print(df['status_group'].value_counts())
print(sns.countplot(x='status_group', data=df))
print(df.dtypes)
print(df.isnull().value_counts())


# Check for irregularities in each column
def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column} : {df[column].unique()}')


print(print_unique_col_values(df))

# Separate the target variable and the dependant features

X = df.drop(['num_private', 'status_group'], axis=1)
y = df['status_group']
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
print(X_train.shape)
print(y_train.shape)

# Subplots of the numerical data
col = ['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population']
plt.figure(figsize=(30, 15))
list(enumerate(col))
for i in enumerate(col):
    plt.subplot(2, 3, i[0] + 1)
    X_train[i[1]].plot.hist(bins=20)
plt.show()

# Function to impute missing values
cateogry_columns = df.select_dtypes(include=['object']).columns.tolist()
integer_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()


def missing_val_imputer(input_data):
    for column in input_data.columns:
        input_data[column].fillna(input_data[column].mode()[0], inplace=True)


# Buiding a transformer for numeric values and onehot enconder for cateorical values
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
# missing_values = Pipeline(steps=[('imputer', CategoricalImputer())])
# Combine both transformers into one pipeline
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, selector(dtype_include=['float64', 'int64'])),
                                               ('cat', cat_transformer, selector(dtype_include='object'))])

# Build a neural network classifier using MLP from Sklearn
model = MLPClassifier(hidden_layer_sizes=[6, 2], batch_size=100, activation='relu', solver='sgd',
                      learning_rate_init=0.001, early_stopping=True, n_iter_no_change=200)

# Training the model by combining the preprocessing step and model building
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
clf.fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean())
print("Accuracy score the test set: ", clf.score(X_test, y_test))

# Saving the model to disk
with open("task1.pickle", "wb") as f:
    pickle.dump(clf, f)

# Working with the test dataset
pickle_in = open("task1.pickle", "rb")
model = pickle.load(pickle_in)
df_test = load_dataset('task1_test.csv')
X_test1 = df_test.drop(['num_private', 'status_group'], axis=1)
y_test1 = df_test['status_group']
print(model.predict(X_test1))
print("Accuracy score the test set: ", model.score(X_test1, y_test1))

# Working with the test dataset without the target label
df_test_no_label = load_dataset('task1_test_nolabels.csv')
X_test_no_label = df_test_no_label.drop('num_private', axis=1)
# print(model.predict(X_test_no_label))
predict = model.predict(X_test_no_label)
result = []
for x in range(len(predict)):
    result.append(predict[x])
X_test_no_label['status_group'] = result
save = X_test_no_label.drop(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population'], axis=1)
save.to_csv('task1_repair.csv', index=False)


# Task2
df_task2 = load_dataset('task2_train.csv')
print(df_task2.sample(10))
print(df_task2.isna().sum())
print_unique_col_values(df_task2)
print(df_task2.shape)
#missing_val_imputer(df_task2)
#print(df_task2.isna().sum())

# preprecessing data for task 2
X_task2 = df_task2.drop(['num_private', 'date_recorded', 'scheme_name', 'recorded_by', 'status_group',
                         'lga', 'waterpoint_type_group', 'source_class', 'quantity_group', 'quality_group',
                         'payment', 'extraction_type_class', 'extraction_type_group'], axis=1)
y_task2 = df_task2['status_group']
cat_transformer = Pipeline(steps=[('missing values', missing_val_imputer(X_task2))
    , ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor_task2 = ColumnTransformer(
    transformers=[('num', num_transformer, selector(dtype_include=['float64', 'int64'])),
                  ('cat', cat_transformer, selector(dtype_include='object'))])

# Separate the target variable and the dependant features
print(X_task2.shape)
print(y_task2.shape)
X_train, X_test, y_train, y_test = train_test_split(X_task2, y_task2, test_size=0.3, shuffle=True, random_state=42)
print(X_train.shape)
print(y_train.shape)

# Build a neural network classifier using MLP from Sklearn
model_task2 = MLPClassifier(hidden_layer_sizes=[28, 2], batch_size=100, activation='relu', solver='sgd',
                            learning_rate_init=0.001, early_stopping=True, n_iter_no_change=200)

# Training the model by combining the preprocessing step and model building (Task 2)
clf_task2 = Pipeline(steps=[('preprocessor', preprocessor_task2), ('classifier', model_task2)])
clf_task2.fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean())
print("Accuracy score the task2: ", clf_task2.score(X_test, y_test))

# Saving the model to disk task 2
with open("task2.pickle", "wb") as f:
    pickle.dump(clf_task2, f)

# Working with the test dataset
pickle_in = open("task2.pickle", "rb")
model = pickle.load(pickle_in)
df_test2 = load_dataset('task2_test.csv')
X_test2 = df_test2.drop(['num_private', 'date_recorded', 'scheme_name', 'recorded_by', 'status_group', 'lga',
                         'waterpoint_type_group', 'source_class', 'quantity_group', 'quality_group', 'payment',
                         'extraction_type_class', 'extraction_type_group'], axis=1)
y_test2 = df_test2['status_group']
print(model.predict(X_test2))
print("Accuracy score the task2 for test dataset: ", model.score(X_test2, y_test2))

# Working with the test dataset without the target label task2
df_test_no_label = load_dataset('task2_test_nolabels.csv')
X_test_no_label = df_test_no_label.drop(['num_private', 'date_recorded', 'scheme_name', 'recorded_by','lga',
                                         'waterpoint_type_group', 'source_class', 'quantity_group', 'quality_group',
                                         'payment', 'extraction_type_class', 'extraction_type_group'], axis=1)

# print(model.predict(X_test_no_label))
predict = model.predict(X_test_no_label)
result = []
for x in range(len(predict)):
    result.append(predict[x])
X_test_no_label['status_group'] = result
print(result)
# save = X_test_no_label.drop(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population'], axis=1)
# save.to_csv('task2_repair.csv', index=False)


# Implementation of task 3
df_task3 = load_dataset('task3_train.csv')
print(df_task3.sample(10))
print(df_task3.isna().sum())
print_unique_col_values(df_task3)
print(df_task3.shape)
print(df_task3['status_group'].value_counts())

# preprecessing data for task 3
X_task3 = df_task3.drop(['num_private', 'date_recorded', 'lga', 'waterpoint_type_group', 'recorded_by', 'status_group',
                         'source_class', 'quantity_group', 'quality_group', 'payment', 'extraction_type_class',
                         'extraction_type_group', 'scheme_name'], axis=1)
y_task3 = df_task3['status_group']
cat_transformer = Pipeline(steps=[('missing values', missing_val_imputer(X_task3))
    , ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor_task3 = ColumnTransformer(
    transformers=[('num', num_transformer, selector(dtype_include=['float64', 'int64'])),
                  ('cat', cat_transformer, selector(dtype_include='object'))])


# Separate the target variable and the dependant features
print(X_task3.shape)
print(y_task3.shape)
X_train, X_test, y_train, y_test = train_test_split(X_task3, y_task3, test_size=0.3, shuffle=True, random_state=42)
print(X_train.shape)
print(y_train.shape)

# Build a neural network classifier using MLP from Sklearn
model_task3 = MLPClassifier(hidden_layer_sizes=[28, 2], batch_size=100, activation='relu', solver='sgd',
                            learning_rate_init=0.001, early_stopping=True, n_iter_no_change=200)

# Training the model by combining the preprocessing step and model building (Task 2)
clf_task3 = Pipeline(steps=[('preprocessor', preprocessor_task3), ('classifier', model_task3)])
clf_task3.fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean())
print("Accuracy score the task3: ", clf_task3.score(X_test, y_test))

# Saving the model to disk task 3
with open("task3.pickle", "wb") as f:
    pickle.dump(clf_task3, f)

# Working with the test dataset
pickle_in = open("task3.pickle", "rb")
model = pickle.load(pickle_in)
df_test3 = load_dataset('task3_test.csv')
X_test3 = df_test3.drop(['num_private', 'date_recorded', 'scheme_name', 'recorded_by', 'status_group', 'lga',
                         'waterpoint_type_group', 'source_class', 'quantity_group', 'quality_group', 'payment',
                         'extraction_type_class', 'extraction_type_group'], axis=1)
y_test3 = df_test3['status_group']
print(model.predict(X_test3))
print("Accuracy score the task3: ", model.score(X_test3, y_test3))

# Working with the test dataset without the target label task2
df_test_no_label = load_dataset('task3_test_nolabels.csv')
X_test_no_label = df_test_no_label.drop(['num_private', 'date_recorded', 'scheme_name', 'recorded_by','lga',
                                         'waterpoint_type_group', 'source_class', 'quantity_group', 'quality_group',
                                         'payment', 'extraction_type_class', 'extraction_type_group'], axis=1)

# print(model.predict(X_test_no_label))
predict = model.predict(X_test_no_label)
result = []
for x in range(len(predict)):
    result.append(predict[x])
X_test_no_label['status_group'] = result
print(result)
# save = X_test_no_label.drop(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'population'], axis=1)
# save.to_csv('task2_repair.csv', index=False)