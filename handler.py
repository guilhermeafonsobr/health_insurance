import pickle
import os
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance

# load model
model = pickle.load(open('model/xgbclassifier_model.pkl', 'rb'))

# initialize API
app = Flask(__name__)

@app.route('/healthinsurance/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()
    
    if test_json: #there is data
        if isinstance(test_json, dict): #unique row
            test_raw = pd.DataFrame(test_json, index = [0])
        
        else: # multiple rows
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
        
        test_raw_copy = test_raw.copy()
        
        # instantiate HealthInsurance class
        pipeline = HealthInsurance()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        print('apos df1')
        print(test_raw.head())
        print(test_raw.head().values)
        
        # feature engineering
        df3 = pipeline.feature_engineering(df1)
        print('apos df3')
        print(test_raw.head())
        print(test_raw.head().values)

        # data preparation
        df4 = pipeline.data_preparation(df3)
        print('apos df4')
        print(test_raw.head())
        print(test_raw.head().values)
           
        # prediction    
        df_response = pipeline.get_prediction(model, test_raw_copy, df4)
        
        return df_response
    
    else:
        return Response('{}', status = 200, mimetype = 'application/json')
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
