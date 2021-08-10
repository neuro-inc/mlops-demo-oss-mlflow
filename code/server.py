import predict
import torch
from sanic import Sanic
from sanic.response import json


app = Sanic("n")
rnn = torch.load('results//model.pt')


@app.route('/')
async def index(request):
    input_line = request.json['line'] if request.json else ''
    predictions = predict.perform(rnn, input_line)
    return json({'predictions': predictions})


@app.route('/test')
async def index(request):
    return json({'hello': 'world'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True)
