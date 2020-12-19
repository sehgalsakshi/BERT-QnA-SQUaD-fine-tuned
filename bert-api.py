from flask import Flask, request
from flask_cors import CORS, cross_origin
import ModelPrediction as model

app = Flask(__name__)
CORS(app)

@app.route('/fetch_answer/', methods=['POST'])
def fetch_answer():
    try:
        context = request.json['paragraph']
        question = request.json['question']
        return model.predict(context, question)
    except:
        print(e)
        return 'Error proccessing your request', 422
    
if __name__ == '__main__':
    app.run('127.0.0.1',port=5000,debug=False)
