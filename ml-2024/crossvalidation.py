import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np

train_data = pd.read_csv('ml-2024-f/train_final.csv')
test_data = pd.read_csv('ml-2024-f/test_final.csv')

X_train = train_data.drop(columns=['income>50K'])
y_train = train_data['income>50K']
X_test = test_data.drop(columns=['ID'])

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train['capital.gain'] = np.log1p(X_train['capital.gain'])
X_train['capital.loss'] = np.log1p(X_train['capital.loss'])
X_test['capital.gain'] = np.log1p(X_test['capital.gain'])
X_test['capital.loss'] = np.log1p(X_test['capital.loss'])

bins = [0, 25, 35, 45, 55, 65, 100]
labels = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
X_train['age_binned'] = pd.cut(X_train['age'], bins=bins, labels=labels)
X_test['age_binned'] = pd.cut(X_test['age'], bins=bins, labels=labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_columns.append('age_binned')
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

X_train_num_df = pd.DataFrame(X_train_scaled, columns=numerical_columns, index=X_train.index)
X_test_num_df = pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index)
X_train_cat_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)

X_train_final = pd.concat([X_train_num_df, X_train_cat_df], axis=1)
X_test_final = pd.concat([X_test_num_df, X_test_cat_df], axis=1)

model = GradientBoostingClassifier()

param_grid = {
    'n_estimators': [350, 400, 450],
    'learning_rate': [0.112, 0.115, 0.12],
    'max_depth': [5, 6],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=make_scorer(roc_auc_score),
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation AUC Score:", grid_search.best_score_)

predictions = best_model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({'ID': test_data['ID'], 'income>50K': predictions})
submission.to_csv('submission.csv', index=False)
