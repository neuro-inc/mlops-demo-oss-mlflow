import json
import glob
import os
import string
import unicodedata
from pathlib import Path


def get_letters():
    return string.ascii_letters + " .,;'"

def get_categories(dirname):
    categories = []
    for filename in findFiles(dirname / "names" / "*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        categories.append(category)
    return categories

def save_categories(categories, filename='results//categories.txt'):
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
