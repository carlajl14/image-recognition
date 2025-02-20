from flask import Flask
from predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    return predict()

if __name__ == '__main__':
    app.run(debug=False)
