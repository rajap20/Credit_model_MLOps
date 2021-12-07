!pip install wandb
!pip install onnxruntime
!pip install -q gradio

import os
import wandb
import pandas as pd
import numpy as np
import onnxruntime as rt
from gradio import gradio as gr

"""## Initialize Weights and Biases"""

os.environ["WANDB_API_KEY"] = "f38031d4fe33eb56cb0d138fc5da4115578dbf43"

run = wandb.init(project='creditpredict')

"""## Download the model and load it"""

ARTIFACT_NAME = 'XGBoost_credit:v2'

#run = wandb.init()
artifact = run.use_artifact('rajap20/creditpredict/' + ARTIFACT_NAME, type='model')
artifact_dir = artifact.download()

#run = wandb.init()ARTIFACT_NAME
#artifact = run.use_artifact('rajap20/creditpredict/' + , type='model')
#artifact_dir = artifact.download()

artifact_dir

!ls -al  ./artifacts/XGBoost_credit:v2

"""http://onnx.ai/sklearn-onnx/auto_examples/plot_complex_pipeline.html#example-complex-pipeline

## Columns
"""

x_columns = ['area_code', 
              'state', 'resid_type', 'net_irr', 'proc_fee',
       'asset_cost', 'loan_amt', 'emi', 'net_salary', 'roi', 'tenure',
       'age']

cat_features = ['area_code', 
                'state', 'resid_type']

num_features = list(set(x_columns) - set(cat_features))

"""## Implement the predict() function

The followng inputs need to be supplied:

['KM_Driven', 'Fuel_Type', 'age',
'Transmission', 'Owner_Type', 'Seats',
'make', 'mileage_new', 'engine_new', 'model',
'power_new', 'Location']
"""

def predict_price(area_code, state, resid_type, net_irr, proc_fee,
       asset_cost, loan_amt, emi, net_salary, roi, tenure, age):


    inputs_dict = {'area_code' : area_code,
                  'state' :  state,
                  'resid_type' :  resid_type,
                  'net_irr' :  float(net_irr),
                  'proc_fee' : float(proc_fee),
                  'asset_cost' : float(asset_cost),
                  'loan_amt' :  float(loan_amt),
                  'emi' :  float(emi),                 
                  'net_salary' : float(net_salary),
                  'tenure' : float(tenure),
                  'roi' : float(roi),
                  'age' : float(age)}

    df = pd.DataFrame(inputs_dict, index = [0])
    print(df)

    inputs = {c: df[c].values for c in df.columns}
    for c in num_features:
        inputs[c] = inputs[c].astype(np.float32)
    for k in inputs:
        inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))            
  
    sess = rt.InferenceSession(artifact_dir + '/credit_xgboost.onnx')
    pred_onx = sess.run(None, inputs)

    predicted_price = float(pred_onx[1][0][1])
    return {f'Probability of default: {np.round(predicted_price, 2)}'}

"""## Implement UI"""

area_code = gr.inputs.Textbox(default='3075.0', label="Area Code")
state = gr.inputs.Dropdown(list(["AP", "AS", "BR", "CG", "DL", "UP", "HA", "GJ", "HP", "CH", "JH",
       "JK", "KA", "KL", "MH", "MP", "OR", "TN", "PY", "PB", "RJ", "WB",
      "UC", "TR", "MN"]), default="AP", label="State")
tenure = gr.inputs.Number(default=0.470588	, label="Tenure")
roi = gr.inputs.Number(default=-0.793282, label="Rate of Interest")
emi = gr.inputs.Number(default=0.124252, label="EMI")
proc_fee = gr.inputs.Number(default=-0.093758, label="Processing Fee")
asset_cost = gr.inputs.Number(default=87000.0, label="Asset cost")
loan_amt = gr.inputs.Number(default=71000.0, label="Loan Amount")
resid_type = gr.inputs.Radio(["O", "R", "L"],  default="O", label = "Residence Type (O: Owned, R: Rented, L: Leased)")
age = gr.inputs.Number(default=0.306122, label="Age")
net_salary = gr.inputs.Number(default=	0.158169, label="Net Salary")
net_irr = gr.inputs.Number(default=-1.321153, label="Net Internal Rate of return")

gr.Interface(predict_price, [area_code, state, resid_type, net_irr, proc_fee, asset_cost, loan_amt, emi, net_salary, roi, tenure, age], "text", live=False).launch(debug=True);
