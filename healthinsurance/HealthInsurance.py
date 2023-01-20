import pickle
import numpy as np
import pandas as pd

class HealthInsurance:
    def __init__(self): 
        # annual premium
        self.annual_premium_scaler = pickle.load(open('parameter/annual_premium_scaler.pkl', 'rb'))
        
        # age scaler
        self.age_scaler = pickle.load(open('parameter/age_scaler.pkl', 'rb'))
        
        # region code
        self.target_encode_region_code_scaler = pickle.load(open('parameter/target_encode_region_code.pkl', 'rb'))
        
        # policy sales channel
        self.fe_policy_sales_channel_scaler = pickle.load(open('parameter/fe_policy_sales_channel.pkl', 'rb'))

        # vintage
        # this feature was not chosen
        
   
    def data_cleaning(self, df1):
        #columns_name = ['id', 'gender', 'age', 'region_code', 'policy_sales_channel','driving_license', 
        #                'vehicle_age', 'vehicle_damage','previously_insured', 'annual_premium', 'vintage']
        
        columns_name = ['id', 'gender', 'age', 'driving_license','region_code','previously_insured',
                        'vehicle_age', 'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage']
        
        df1.columns = columns_name
        
        return df1
        
        
    def feature_engineering (self, df3):
        
        # vehicle damage 
        dict_vehicle_damage = {'Yes':1, 'No':0}
        df3['vehicle_damage'] = df3['vehicle_damage'].map(dict_vehicle_damage)
        
        # vehicle age (fazer dessa forma pode dar problema se rodar o snipet mais de uma vez)
        df3['vehicle_age'] = df3['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year')
        
        return df3
    
        
    def data_preparation (self, df4):
        
        
        # annual_premium
        df4['annual_premium'] = self.annual_premium_scaler.transform(df4[['annual_premium']].values)

        # age
        df4['age'] = self.age_scaler.transform(df4[['age']].values)

        # region_code
        df4['region_code'] = df4['region_code'].map(self.target_encode_region_code_scaler)

        # policy_sales_channel
        df4['policy_sales_channel'] = df4['policy_sales_channel'].map(self.fe_policy_sales_channel_scaler)
        
        # vehicle_age
        df4 = pd.get_dummies(df4, prefix='vehicle_age', columns = ['vehicle_age'])
        
        # gender
        df4 = pd.get_dummies(df4, prefix='gender', columns = ['gender'])
        
        # vintage
        # this feature was not chosen
        
        # as gender and age are derived in multiple columns (due to pandas get_dummies method) it is necessary to perform a check:

        if "gender_Male" not in df4.columns:
            df4["gender_Male"] = 0
        
        if "gender_Female" not in df4.columns:
            df4["gender_Female"] = 0
            
        if "vehicle_age_below_1_year" not in df4.columns:
            df4["vehicle_age_below_1_year"] = 0
        
        if "vehicle_age_between_1_2_year" not in df4.columns:
            df4["vehicle_age_between_1_2_year"] = 0
        
        if "vehicle_age_over_2_years" not in df4.columns:
            df4["vehicle_age_over_2_years"] = 0
        
        cols_selected_boruta = ['age','region_code', 'policy_sales_channel', 'vehicle_damage', 'previously_insured', 
                                'annual_premium','vehicle_age_below_1_year', 'vehicle_age_between_1_2_year',
                                'vehicle_age_over_2_years', 'gender_Female', 'gender_Male']
            
        return df4[cols_selected_boruta]
    
    def get_prediction(self, model, original_data, test_data):
        
        # model prediciton
        pred = model.predict_proba(test_data)
        
        # creating a prediction dataframe of the predict_proba (class 0 and 1 predictions)
        table_proba = pd.DataFrame(pred)
        
        # join prediction into original data
        original_data['Score'] = table_proba[1]
        
        original_data.sort_values('Score', ascending = False, inplace = True)
        
        return original_data.to_json(orient = 'records', date_format = 'iso')
