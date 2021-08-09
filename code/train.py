import datetime
import math
import mlflow
import os
import platform
import time
import train
import torch
import torch.nn as nn
import utils
from torch.utils.tensorboard import SummaryWriter
from mlflow.entities import Param, Metric

import plot
import data
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
        writer.add_scalar(f"training/loss", loss, global_step=iter)
        if iter % print_every == 0:
            guess, _ = utils.categoryFromOutput(output, categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    return current_loss / n_iters


def main():

    args = utils.get_args()
    
    # 1. Starting tracking session
    client = mlflow.tracking.MlflowClient()
    # 1. Starting tracking session
    mlflow.set_experiment(args.experiment_name)
    
    # 1. Starting tracking session
    mlrun = mlflow.start_run()
    params = []
    metrics = []
    start = 128

    record_uuid =  str(datetime.datetime.now().timestamp()).replace('.', '')
    record_path = os.path.join('results', record_uuid)

    for i in range(3):
        n_hidden = start + i
        params += [Param(f'n_{i}', str(n_hidden))]

        # # 3. Setting tags
        mlflow.set_tag('Python', platform.python_version())
        mlflow.set_tag('Machine', platform.machine())
        mlflow.set_tag('Node', platform.node())
        mlflow.set_tag('Platform', platform.platform())
        # # 5. Logging parameters
        # mlflow.log_param(f'n_hidden_{n_hidden}', n_hidden)

        letters = data.get_letters()
        category_lines = data.read_files(args.data_path, letters)
        categories = data.get_categories(args.data_path)
        n_letters = len(letters)
        n_categories = len(categories)

        data.save_categories(categories)
        rnn = model.RNN(n_letters, n_hidden, n_categories)
        writer = SummaryWriter(log_dir=args.dump_dir / args.experiment_name)
        loss = perform(rnn, category_lines, categories, letters, args.n_iters, writer)
        # filename = 'results//model.pt'
        # torch.save(rnn, filename)

        # 6. Logging metrics
        now = round(time.time())
        metrics += [Metric("rmse", loss, now, 10)]

        # 7. Saving artefacts
        
        model_path = os.path.join(record_path, f'model_{n_hidden}.pt')
        chart_path = os.path.join(record_path, f'chart_{n_hidden}.png')
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
    client.log_batch(mlrun.info.run_uuid, metrics, params)
    client.log_artifacts(mlrun.info.run_id, record_path)

    # 7. Terminating tracking session
    mlflow.end_run()


if __name__ == '__main__':
    main()

