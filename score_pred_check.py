import streamlit as st
import pandas as pd
import joblib
load_model=joblib.load('score_pred.pkl')
load_scaler=joblib.load('score_scaler.pkl')
st.markdown("# SCORE PREDICTION!")

main_list = []
st.markdown("###### Write down number of hours you study here: ")
hours_study =st.number_input("",key='input_1')
if hours_study:
 main_list.append(hours_study)
st.markdown("###### Write down your previous score here out of 100: ")
prev_score = st.number_input("",key='input_2',format="%d",step=1)
if prev_score:
 main_list.append(prev_score)
st.markdown("###### Do you take part in Extracurricular Activities? ")
Extra_activity = st.radio(
    "",
    ['Choose one','Yes','No']
)
if Extra_activity!='Choose one':
   if Extra_activity == 'Yes':
     main_list.append(1)
   else:
     main_list.append(0)
st.markdown("###### Write down number of hours you sleep here: ")
Sleep_hours = st.number_input('',key='input_3')
if Sleep_hours:
 main_list.append(Sleep_hours)
st.markdown("###### Write down number of past papers you solved here: ")
question_solve = st.number_input('',key='input_4',format="%d",step=1)
if question_solve:
 main_list.append(question_solve)

 main_df = pd.DataFrame([main_list], columns=['Hours Studied', 'Previous Scores', 'Sleep Hours', 
                                             'Sample Question Papers Practiced', 'Extracurricular Activities'])
 change_data = load_scaler.transform(main_df)
 predicting = load_model.predict(change_data)
 real_prediction=round(predicting[0],2)
 st.write(f"#### Here is your predicted score: {real_prediction}")
