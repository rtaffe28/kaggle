import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('ml-2024-f/train_final.csv')

train_data = data[:15000]
test_data = data[15000:]

X_train = train_data.drop(columns=['income>50K'])
y_train = train_data['income>50K']
print(len(X_train.columns))

X_test = test_data.drop(columns=['income>50K'])
y_test = test_data['income>50K']

accs = {}
for feature in X_train.keys():
    print(feature)
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_train = encoder.fit_transform(X_train[categorical_columns])
    encoded_test = encoder.transform(X_test[categorical_columns])

    one_hot_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
    one_hot_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)

    encoded_train_f = pd.concat([X_train.drop(categorical_columns, axis=1), one_hot_train], axis=1)
    encoded_test_f = pd.concat([X_test.drop(categorical_columns, axis=1), one_hot_test], axis=1)
    print(len(encoded_train_f.columns))

    train_feature = encoded_train_f.drop(feature, axis=1)
    test_feature = encoded_test_f.drop(feature, axis=1)
    model = GradientBoostingClassifier()
    model.fit(train_feature, y_train)

    predictions = model.predict(test_feature)

    test_accuracy = accuracy_score(y_test, predictions)
    accs[feature] = test_accuracy

print(sorted(accs.items()))
