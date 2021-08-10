import random
import torch
import json
import glob
import os
import string
import unicodedata
from pathlib import Path


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


def get_letters():
    return string.ascii_letters + " .,;'"


def get_categories(dirname):
    categories = []
    for filename in findFiles(dirname / "names" / "*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        categories.append(category)
    return categories


def save_categories(categories, filename='results//categories.txt'):
    os.makedirs('results', exist_ok=True)
    f = open(filename, "w")
    f.write(json.dumps(categories))
    f.close()


def load_categories(filename='results//categories.txt'):
    f = open(filename, "r")
    text = json.loads(f.read())
    print(text)
    return text


def findFiles(path: Path):
    return glob.glob(str(path))


def unicodeToAscii(s, letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in letters
    )


def read_lines(filename, letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, letters) for line in lines]


def read_files(dirname, letters):
    category_lines = {}
    for filename in findFiles(dirname / "names" / "*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = read_lines(filename, letters)
        category_lines[category] = lines
    return category_lines
