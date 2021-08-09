import argparse
import os
import random
import torch
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model for Names classification with a Character-Level RNN."
    )
    parser.add_argument(
        "--dump_dir",
        type=Path, default="../results", help="Dump path."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default='Default',
        help="Experiment name, defaults to job ID."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default="data",
        help="Path to folder, where training data is located"
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=100000,
        help="Number of iterations to train the model"
    )
    return parser.parse_args()

def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def categoryFromOutput(output, categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i

def letterToIndex(letter, letters):
    return letters.find(letter)

def letterToTensor(letter, letters):
    n_letters = len(letters)
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter, letters)] = 1
    return tensor

def lineToTensor(line, letters):
    n_letters = len(letters)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter, letters)] = 1
    return tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(category_lines, categories, letters):
    category = randomChoice(categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line, letters)
    return category, line, category_tensor, line_tensor
