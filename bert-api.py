from flask import Flask, request
from flask_cors import CORS, cross_origin
from ModelPrediction import *

model = initialize_model()
def create_app():
    app = Flask(__name__)
    return app

app = create_app()
CORS(app)

@app.route('/fetch_answer/', methods=['POST'])
def fetch_answer():
    try:
        context = request.json['paragraph']
        question = request.json['question']
        global model
        answer = predict(context, question, model)
        print(answer)
        return answer
    except Exception as e:
        print(e)
        return 'Error proccessing your request', 422
    
# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"
	
if __name__ == '__main__':
    app.run('127.0.0.1',port=5000,debug=False)
