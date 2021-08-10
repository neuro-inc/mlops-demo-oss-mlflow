import datetime
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
from torch.utils.tensorboard import SummaryWriter
from mlflow.entities import Param, Metric
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


def perform(rnn, category_lines, categories, letters, n_iters, writer):
    print_every=int(n_iters * 0.05)
    current_loss = 0
    start = time.time()
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = utils.randomTrainingExample(category_lines, categories, letters)
        output, loss = train(rnn, category_tensor, line_tensor)
        current_loss += loss
        writer.add_scalar(f'training/loss', loss, global_step=iter)
        if iter % print_every == 0:
            guess, _ = utils.categoryFromOutput(output, categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    return current_loss / n_iters


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
        default=100000,
        help='Number of iterations to train the model'
    )
    return parser.parse_args()


def main():

    args = get_args()
    
    # 1. Setting the experiemnt name
    mlflow.set_experiment(args.experiment_name)
    
    # 2. Starting the tracking session
    mlrun = mlflow.start_run()

    # 3. Initializing experiment layout
    record_uuid =  str(datetime.datetime.now().timestamp()).replace('.', '')
    record_path = os.path.join('results', record_uuid)
    n_hidden = int(args.n_hidden)

    # 4. Setting the tags
    mlflow.set_tag('Python', platform.python_version())
    mlflow.set_tag('Machine', platform.machine())
    mlflow.set_tag('Node', platform.node())
    mlflow.set_tag('Platform', platform.platform())

    # 5. Logging the parameter
    mlflow.log_param(f'n_hidden', n_hidden)

    letters = utils.get_letters()
    category_lines = utils.read_files(args.data_path, letters)
    categories = utils.get_categories(args.data_path)
    n_letters = len(letters)
    n_categories = len(categories)

    utils.save_categories(categories)
    rnn = model.RNN(n_letters, n_hidden, n_categories)
    writer = SummaryWriter(log_dir=args.dump_dir / args.experiment_name)
    loss = perform(rnn, category_lines, categories, letters, args.n_iters, writer)

    # 6. Logging the metric
    mlflow.log_metric('loss', loss)

    # 7. Saving the artefacts
    model_path = os.path.join(record_path, f'model_{n_hidden}.pt')
    chart_path = os.path.join(record_path, f'chart_{n_hidden}.png')
    code_path = os.path.join(record_path, f'code_{n_hidden}')
    cache_path = os.path.join('code', f'__pycache__')

    os.makedirs(record_path, exist_ok=True)
    torch.save(rnn, model_path)
    plot.draw(
        rnn,
        category_lines,
        categories,
        letters,
        10000,
        chart_path
    )
    shutil.rmtree(cache_path, ignore_errors=True)
    shutil.make_archive(code_path, 'zip', 'code')

    # 8. Terminating tracking session
    mlflow.end_run()


if __name__ == '__main__':
    main()
