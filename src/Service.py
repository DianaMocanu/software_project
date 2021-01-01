from flask import Flask, jsonify
from flask import request
from src.Controller import Controller
from flask_cors import CORS, cross_origin
from mysql.connector import Error
import time
from src.QueryValidator import QueryValidator

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources=r'/*', headers='Content-Type')
service = Controller()
validator = QueryValidator()

@app.before_request
def oauth_verify(*args, **kwargs):
    if request.method in ['OPTIONS', ]:
        return


@app.route('/generate', methods=['POST'])
@cross_origin()
def generateQuery():
    request_json = request.get_json()
    Data = request_json['Data']
    database = Data['database']
    query = Data['query']
    negation = Data['negation']
    rate = Data['rate']
    try:
        isValidated, message = validator.checkIntegralQuery(query)
        if not isValidated:
            return message, 209
        start_time = time.time()
        result, positive_ids= service.getQueryAlternativeConditions(query, database, negation, rate)
        print("--- %s seconds ---" % (time.time() - start_time))
        dataToSend = {'results': result, 'pos_ids': positive_ids}
        response = jsonify(dataToSend)
        return response
    except Error as n:
        return n.msg, 209

@app.route('/executeId', methods=['POST'])
@cross_origin()
def executeQueryId():
    request_json = request.get_json()
    Data = request_json['Data']
    database = Data['database']
    query = Data['query']
    try:
        result = service.executeQueryId(query, database)
        dataToSend = {'results': result}
        response = jsonify(dataToSend)
        return response
    except Error as n:
        return n.msg, 209


if __name__ == '__main__':
    app.run()
