# MLFlow Tutorial in Neuro MLOps Platform 

This tutorial shows how to prepare, monitor and test ML pipeline with MLFlow.
The core problem is an adaption of the [NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial to the [Neu.ro platform](https://neu.ro).

## Quick Start

Sign up at [app.neu.ro](https://app.neu.ro) and setup your local machine according to [instructions](https://docs.neu.ro/).

Good examples could be found in the [repository](https://github.com/amesar/mlflow-examples)
 

See [Help.md](HELP.md) for the detailed Neuro Project Template Reference.

Install and log in the platform
```shell
pip install -U neuro-cli neuro-flow neuro-extras
neuro login
```

Look at the live.yml and set the defaults
```yaml
defaults:
  preset: cpu-large
  life_span: 1d
```

Buuild the main image
```shell
neuro-flow build myimage
```

Create the volumes in the platform
```shell
neuro-flow mkvolumes
```

Upload the volumes contents
```shell
neuro-flow upload ALL
```

Run an MLFlow server for experiment and model tracking. In this setup, we affect our github action.

```shell
neuro-flow run mlflow
```

```python
def main():
    args = get_args()
    
    # 1. Set the remote backend uri
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    # 2. Setting the experiemnt name
    mlflow.set_experiment(args.experiment_name)
    
    # 3. Starting the tracking session
    mlflow.start_run()

    # 4. Initializing experiment layout
    rec_uuid =  str(datetime.datetime.now().timestamp()).replace('.', '')
    rec_path = os.path.join('results', rec_uuid)
    n_hidden = int(args.n_hidden)

    # 5. Setting the tags
    mlflow.set_tag('Python', platform.python_version())
    mlflow.set_tag('Machine', platform.machine())
    mlflow.set_tag('Node', platform.node())
    mlflow.set_tag('Platform', platform.platform())
    mlflow.set_tag('Record', rec_uuid)

    # 6. Logging the parameter
    mlflow.log_param(f'n_hidden', n_hidden)

    letters = utils.get_letters()
    category_lines = utils.read_files(args.data_path, letters)
    categories = utils.get_categories(args.data_path)
    n_letters = len(letters)
    n_categories = len(categories)

    utils.save_categories(categories)
    rnn = model.RNN(n_letters, n_hidden, n_categories)
    writer = SummaryWriter(log_dir=args.dump_dir / args.experiment_name)

    # 7. Logging the metric
    loss = perform(rnn, category_lines, categories, letters, args.n_iters, writer)

    # 8. Saving the artefacts
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

    # 9. Logging the artifacts
    mlflow.pytorch.log_model(rnn, 'model')
    mlflow.log_artifacts(rec_path)
    
    # 10. Terminating the tracking session
    print(f'The record {rec_uuid} was created') 
    mlflow.end_run()

```

Run training for number of neurons in the hidden layer n_hidden=120 and see the results in the mlflow dashboard
```shell
neuro-flow run train --param n_hidden 120
```
