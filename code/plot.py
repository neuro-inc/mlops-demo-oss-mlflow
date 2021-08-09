import torch
import utils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


plt.figure()

def draw(rnn, category_lines, categories, letters, n_confusion=10000, filename='results//figure.png'):
    n_categories = len(categories)
    confusion = torch.zeros(n_categories, n_categories)
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = utils.randomTrainingExample(category_lines, categories, letters)

        output = utils.evaluate(rnn, line_tensor)
        guess, guess_i = utils.categoryFromOutput(output, categories)
        category_i = categories.index(category)
        confusion[category_i][guess_i] += 1
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    labels = [''] + categories
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(filename)
