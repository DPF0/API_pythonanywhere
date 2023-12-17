from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

#1
@app.route('/v2/predict', methods=['GET'])
def predict():
    model = pickle.load(open('./data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)


    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[int(tv), int(radio), int(newspaper)]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'

# app.run()

#2
@app.route("/v2/insert_data", methods=['POST'])
def insert_data():
    
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    sales = request.args.get('sales', None)

    if tv is None or radio is None or newspaper is None or sales is None:
        return "Missing args, the input values are needed."
    else:
        connection = sqlite3.connect('./data/advertising.db')
        cursor = connection.cursor()
        update_db = "INSERT INTO campañas (TV, radio, newspaper, sales) VALUES (?, ?, ?, ?)"
        cursor.execute(update_db, (tv, radio, newspaper, sales,))
        connection.commit()
        connection.close()
        
        return 'Database updated succesfully'

# app.run()


#3
@app.route('/v2/retrain', methods=['PUT'])
def retrain():
    model = pickle.load(open('./data/advertising_model','rb'))
    
    connection = sqlite3.connect('./data/advertising.db')
    cursor = connection.cursor()
    
    query_all = '''SELECT * FROM campañas'''
    
    df = pd.DataFrame(data= cursor.execute(query_all).fetchall(), columns= ['TV', 'radio', 'newspaper', 'sales'])
    
    connection.close()
    
    X = df.drop(columns= 'sales')
    Y = df['sales']
    
    mae1 = mean_absolute_error(model.predict(X), Y)
    
    model.fit(X, Y)
    
    mae2 = mean_absolute_error(model.predict(X), Y)
    
    if mae1 <= mae2:
        return 'No changes made'
    else:
        pickle.dump(model, open('./data/advertising_model', 'wb'))
        return 'Model retrained and saved'


# app.run()