import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv('ml-2024-f/train_final.csv')
test_data = pd.read_csv('ml-2024-f/test_final.csv')

X_train = train_data.drop(columns=['income>50K'])
y_train = train_data['income>50K']
X_test = test_data.drop(columns=['ID'])

categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

mlp = MLPClassifier(max_iter=300, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', mlp)
])

param_grid = {
    'classifier__hidden_layer_sizes': [(100,), (128, 64), (256, 128, 64)],
    'classifier__alpha': [1e-4, 1e-3, 1e-2],
    'classifier__learning_rate_init': [0.001, 0.01],
    'classifier__activation': ['relu', 'tanh']
}

grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best AUC on training data: {grid_search.best_score_:.4f}")

y_test_proba = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'ID': test_data['ID'], 'income>50K': y_test_proba})
submission.to_csv('submission.csv', index=False)
