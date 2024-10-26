import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('ml-2024-f/train_final.csv')
test_data = pd.read_csv('ml-2024-f/test_final.csv')

X_train = train_data.drop(columns=['income>50K'])
y_train = train_data['income>50K']
X_test = test_data.drop(columns=['ID'])

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_train = encoder.fit_transform(X_train[categorical_columns])
encoded_test = encoder.transform(X_test[categorical_columns])

one_hot_train = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
one_hot_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)

encoded_train_f = pd.concat([X_train.drop(categorical_columns, axis=1), one_hot_train], axis=1)
encoded_test_f = pd.concat([X_test.drop(categorical_columns, axis=1), one_hot_test], axis=1)

clf1 = GradientBoostingClassifier()
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = AdaBoostClassifier(n_estimators=200)

voting_clf = VotingClassifier(estimators=[
    ('gb', clf1), 
    ('rf', clf2), 
    ('lr', clf3)],
    voting='soft' 
)

voting_clf.fit(encoded_train_f, y_train)

proba_predictions = voting_clf.predict_proba(encoded_test_f)[:, 1] 

submission = pd.DataFrame({
    'ID': test_data['ID'], 
    'Prediction': proba_predictions
})

submission.to_csv('submission.csv', index=False)
