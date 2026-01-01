import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print("\nMissing values in training set:")
print(train_df.isnull().sum())

# Feature Engineering
def preprocess_data(df, is_train=True):
    """Preprocess and engineer features for the dataset"""
    df = df.copy()
    
    # Fill missing Age values with median by Pclass and Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fill remaining Age NaN with overall median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fill missing Fare with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create Title feature from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
                   'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Create FamilySize feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Create Age bands
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Create Fare bands
    df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'],
                             duplicates='drop')
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    le_title = LabelEncoder()
    df['Title'] = le_title.fit_transform(df['Title'])
    
    le_ageband = LabelEncoder()
    df['AgeBand'] = le_ageband.fit_transform(df['AgeBand'])
    
    le_fareband = LabelEncoder()
    df['FareBand'] = le_fareband.fit_transform(df['FareBand'])
    
    return df

# Preprocess data
train_processed = preprocess_data(train_df, is_train=True)
test_processed = preprocess_data(test_df, is_train=False)

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
            'Title', 'FamilySize', 'IsAlone', 'AgeBand', 'FareBand']

X_train = train_processed[features]
y_train = train_processed['Survived']
X_test = test_processed[features]

print("\n" + "="*50)
print("Feature Importance Analysis")
print("="*50)

# Train Random Forest with default parameters first
rf_basic = RandomForestClassifier(n_estimators=100, random_state=42)
rf_basic.fit(X_train, y_train)

# Cross-validation score
cv_scores = cross_val_score(rf_basic, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nBasic Random Forest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_basic.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance.to_string(index=False))

# Hyperparameter tuning
print("\n" + "="*50)
print("Hyperparameter Tuning with GridSearchCV")
print("="*50)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', 
                          n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_rf = grid_search.best_estimator_
train_accuracy = best_rf.score(X_train, y_train)
print(f"Training accuracy: {train_accuracy:.4f}")

# Make predictions
predictions = best_rf.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})

submission.to_csv('random_forest_submission.csv', index=False)
print("\n" + "="*50)
print("Submission file created: random_forest_submission.csv")
print(f"Total predictions: {len(submission)}")
print(f"Predicted survivors: {predictions.sum()}")
print(f"Predicted non-survivors: {len(predictions) - predictions.sum()}")
print(f"Survival rate: {predictions.mean():.2%}")
print("="*50)
