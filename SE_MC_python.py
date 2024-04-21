import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from src import mispredict_label, BGMD
from parameters import * # coverage and constants in rule learning

# settings
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

data_path = 'data/se/data_Python.csv'
label_path = 'data/se/label_Python.csv'

x = pd.read_csv(data_path)
y = pd.read_csv(label_path)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#print(x.head())

# Decision tree model
model_DT = DecisionTreeClassifier()
scores_DT = cross_val_score(model_DT, X=x_train, y=y_train, cv=FOLDS)
model_DT.fit(x_train, y_train)
y_pred_DT = model_DT.predict(x_test)
#print(classification_report(y_test, y_pred_DT))

# set up dataframes
df_DT_results = pd.concat([x_test, y_test], axis=1, join="inner")
df_DT_results.loc[:,"pred"] = y_pred_DT
#print(df_DT_results)

df_DT_flagged = mispredict_label.mispredict_label(df_DT_results, 'is_conflict', 'pred')
#print(df_DT_flagged)

#print("number of mispredicted data")
#print(len(df_DT_flagged[df_DT_flagged['mispredict']==1]))

test_ruleset = BGMD.BGMD(df_DT_flagged, alpha, lamb1, lamb2, lamb3, delta)
print(test_ruleset)



