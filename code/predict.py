import torch
import utils
import sys


def predict(rnn, input_line, categories, letters, n_predictions=3):
    predictions = []
    with torch.no_grad():
        output = utils.evaluate(rnn, utils.lineToTensor(input_line, letters))
        topv, topi = output.topk(n_predictions, 1, True)
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append({categories[category_index]: value})
    return predictions


def perform(rnn, input_line):
    letters = utils.get_letters()
    categories = utils.load_categories()
    predictions = predict(rnn, input_line, categories, letters)
    return predictions


if __name__ == '__main__':
    if len(sys.argv) > 1:
      input_line = sys.argv[1]
      rnn = torch.load('results//model.pt')
      predictions = perform(rnn, input_line)
      print(predictions)
