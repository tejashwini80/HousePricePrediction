import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)  #starting of app
## Loading the model

# Open the pickle file in binary mode
with open('regmodel.pkl', 'rb') as file:
    regmodel = pickle.load(file)

#regmodel=pickle.load((open('regmodel.pkl','r')))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
  #data=request.get_json(force=True)
  #data=np.array(data)
  #data=data.reshape(1,-1)
  #prediction=regmodel.predict(data)
  #output=prediction[0]
  #return jsonify(output)
  data=request.json['data']    #data is a key
  print(data)
  print(np.array(list(data.values())).reshape(1,-1))
  #standardization
  new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
  output=regmodel.predict(new_data)
  print(output[0])
  return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
   data=[float (x) for x in request.form.values()]
   final_input=scaler.transform(np.array(data).reshape(1,-1))
   print(final_input)
   output=regmodel.predict(final_input)[0]
   return render_template("home.html",prediction_text="The Predicted house price is {}".format(output))



if __name__=="__main__":
  app.run(debug=True)

