import datetime
import json
import math
import mlflow
import os
import platform
import shutil
import time
import train
import torch
import torch.nn as nn
import utils
import argparse
from pathlib import Path

import plot
import model
import utils


def train(rnn, category_tensor, line_tensor, learning_rate=0.005, criterion=nn.NLLLoss()):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def iterate(rnn, category_lines, categories, letters, n_iters):
    print_every=int(n_iters * 0.05)
    current_loss = 0
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = utils.randomTrainingExample(category_lines, categories, letters)
        output, loss = train(rnn, category_tensor, line_tensor)
        current_loss += loss
        if iter % print_every == 0:
            guess, _ = utils.categoryFromOutput(output, categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            mlflow.log_metric('loss', loss)
    return current_loss


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train model for Names classification with a Character-Level RNN.'
    )
    parser.add_argument(
        '--dump_dir',
        type=Path, default='../results', help='Dump path.'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Default',
        help='Experiment name, defaults to job ID.'
    )
    parser.add_argument(
        '--mlflow_uri',
        type=str,
        default='http//localhost:5000',
        help='Mlflow URI.'
    )
    parser.add_argument(
        '--n_hidden',
        type=str,
        default='128',
        help='Number of hidden neurons, defaults to job ID.'
    )
    parser.add_argument(
        '--data_path',
        type=Path,
        default='data',
        help='Path to folder, where training data is located'
    )
    parser.add_argument(
        '--n_iters',
        type=int,
        default=10000,
        help='Number of iterations to train the model'
    )
    return parser.parse_args()


def main():
    args = get_args()
    
    # Setting the remote backend uri
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    # Setting the experiemnt name
    mlflow.set_experiment(args.experiment_name)
    
    # Starting the tracking session
    mlflow.start_run()

    # Initializing experiment layout
    rec_uuid =  str(datetime.datetime.now().timestamp()).replace('.', '')
    rec_path = os.path.join('results', rec_uuid)
    n_hidden = int(args.n_hidden)

    # Setting the tags
    mlflow.set_tag('Python', platform.python_version())
    mlflow.set_tag('Machine', platform.machine())
    mlflow.set_tag('Node', platform.node())
    mlflow.set_tag('Platform', platform.platform())
    mlflow.set_tag('Record', rec_uuid)

    # Logging the parameter
    mlflow.log_param(f'n_hidden', n_hidden)

    letters = utils.get_letters()
    category_lines = utils.read_files(args.data_path, letters)
    categories = utils.get_categories(args.data_path)
    n_letters = len(letters)
    n_categories = len(categories)

    utils.save_categories(categories)
    rnn = model.RNN(n_letters, n_hidden, n_categories)

    # Logging the metric
    loss = iterate(rnn, category_lines, categories, letters, args.n_iters)

    # Saving the artefacts
    info = {
        'uuid': rec_uuid,
        'params': {
            'n_hidden': n_hidden
        },
        'metrics': {
            'cumulative_loss': str(loss)
        },
        'artifacts': [
            'model.pt',
            'chart.png',
            'code.zip'
        ]
    }
    model_path = os.path.join(rec_path, 'model.pt')
    chart_path = os.path.join(rec_path, 'chart.png')
    code_path = os.path.join(rec_path, 'code')
    cache_path = os.path.join('code', '__pycache__')
    info_path = os.path.join(rec_path, 'info.json')

    os.makedirs(rec_path, exist_ok=True)
    torch.save(rnn, model_path)
    plot.draw(
        rnn,
        category_lines,
        categories,
        letters,
        100,
        chart_path
    )
    shutil.rmtree(cache_path, ignore_errors=True)
    shutil.make_archive(code_path, 'zip', 'code')
    with open(info_path, 'w') as info_file:
        json.dump(info, info_file,  indent=4)

    # Logging the artifacts
    mlflow.pytorch.log_model(rnn, 'model')
    mlflow.log_artifacts(rec_path)
    
    # Terminating the tracking session
    print(f'The record {rec_uuid} was created') 
    mlflow.end_run()


if __name__ == '__main__':
    main()
