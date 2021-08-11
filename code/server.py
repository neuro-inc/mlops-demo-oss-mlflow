import argparse
import os
import predict
import torch
from sanic import Sanic
from sanic.response import json
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train model for Names classification with a Character-Level RNN.'
    )
    parser.add_argument(
        '--data_path',
        type=Path,
        default='data',
        help='Path to folder, where training data is located'
    )
    parser.add_argument(
        '--dump_dir',
        type=Path,
        default='results',
        help='Dump path.'
    )
    parser.add_argument(
        '--rec_uuid',
        type=str,
        help='Record uuid to be inerenced with'
    )
    return parser.parse_args()


args = get_args()
app = Sanic("inference_server")
net = torch.load(os.path.join(args.dump_dir, args.rec_uuid, 'model.pt'))


@app.route('/test')
async def index(request):
    return json({'server': 'works'})


@app.route('/')
async def index(request):
    input_line = request.json['line'] if request.json else ''
    predictions = predict.perform(net, input_line)
    return json({'predictions': predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True)
