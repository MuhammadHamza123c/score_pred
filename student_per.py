import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform
import joblib

scaler = MinMaxScaler()
data = pd.read_csv("D:/Code/Student_Performance.csv")
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Extracurricular Activities']]
X_scaled = scaler.fit_transform(X)
y = data['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Ridge()
param_distributions = {
    'alpha': uniform(0.01, 100),
    'solver': ['auto', 'svd', 'lsqr'],
    'fit_intercept': [True, False]
}
random_search = RandomizedSearchCV(model, param_distributions=param_distributions, cv=5, random_state=42)
random_search.fit(X_train, y_train)

prediction = random_search.predict(X_test)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
save_model=joblib.dump(random_search,"score_pred.pkl")
save_scaler=joblib.dump(scaler,'score_scaler.pkl')
# st.markdown("# SCORE PREDICTION!")

# main_list = []
# st.markdown("###### Write down number of hours you study here: ")
# hours_study =st.number_input("",key='input_1')
# if hours_study:
#  main_list.append(hours_study)
# st.markdown("###### Write down your previous score here: ")
# prev_score = st.number_input("",key='input_2',format="%d",step=1)
# if prev_score:
#  main_list.append(prev_score)
# st.markdown("###### Do you take part in Extracurricular Activities? ")
# Extra_activity = st.radio(
#     "",
#     ['Choose one','Yes','No']
# )
# if Extra_activity!='Choose one':
#    if Extra_activity == 'Yes':
#      main_list.append(1)
#    else:
#      main_list.append(0)
# st.markdown("###### Write down number of hours you sleep here: ")
# Sleep_hours = st.number_input('',key='input_3')
# if Sleep_hours:
#  main_list.append(Sleep_hours)
# st.markdown("###### Write down number of past papers you solved here: ")
# question_solve = st.number_input('',key='input_4',format="%d",step=1)
# if question_solve:
#  main_list.append(question_solve)

#  main_df = pd.DataFrame([main_list], columns=['Hours Studied', 'Previous Scores', 'Sleep Hours', 
#                                              'Sample Question Papers Practiced', 'Extracurricular Activities'])
#  change_data = scaler.transform(main_df)
#  predicting = random_search.predict(change_data)
#  real_prediction=round(predicting[0],2)
#  st.write(f"#### Here is your predicted score: {real_prediction}")
