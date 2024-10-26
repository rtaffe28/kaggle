import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier

train_data = pd.read_csv('ml-2024-f/train_final.csv')
test_data = pd.read_csv('ml-2024-f/test_final.csv')

X_train = train_data.drop(columns=['income>50K'])
y_train = train_data['income>50K']

X_test = test_data.drop(columns=['ID'])


drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_test = X_test.select_dtypes(exclude=['object'])

model = BaggingClassifier()
model.fit(drop_X_train, y_train)

predictions = model.predict(drop_X_test)

submission = pd.DataFrame({
    'ID': test_data['ID'], 
    'income>50K': predictions 
})

submission.to_csv('submission.csv', index=False)