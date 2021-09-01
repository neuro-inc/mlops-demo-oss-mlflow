# MLFlow Configuring in Neuro Platform

This tutorial shows how to prepare, monitor and test ML pipeline with MLFlow.
The core problem is an adaption of the [NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial to the [Neu.ro platform](https://neu.ro). Let's figure out how to add the necessary MLFllow configuration to live.yml

1. Add mlruns volume in the volumes section to make it possible saving artifacts and sqlite data to it

```yaml
mlruns:
  remote: storage:${{ flow.flow_id }}/mlruns
  mount: /usr/local/share/mlruns
```

2. Add mlflow action in the jobs section to run mlflow in your setting by triggering the github action from our repository without configuring it manully
```yaml
mlflow:
  action: gh:neuro-actions/mlflow@v1.17.0
  args:
    backend_store_uri: sqlite:///${{ volumes.mlruns.mount }}/mlflow.db
    default_artifact_root: ${{ volumes.mlruns.mount }}
    volumes: "${{ to_json( [volumes.mlruns.ref_rw] ) }}"
```

3. Mount the mlruns volume in the training job section to save artifacts and sqlite data in your script
```yaml
volumes:
  - $[[ upload(volumes.data).ref_ro ]]
  - $[[ upload(volumes.code).ref_ro ]]
  - $[[ volumes.mlruns.ref_rw ]]
```

4. Pass the mlflow_uri to the training script and parse it to integrate your script execution with the mlflow running on our platform
```yaml
bash: |
python -u $[[ volumes.code.mount ]]/train.py \
  --data_path $[[ volumes.data.mount ]] \
  --mlflow_uri http://${{ inspect_job('mlflow').internal_hostname_named }}:5000 \
  --n_hidden ${{ params.n_hidden }}
```

```python
parser.add_argument(
    '--mlflow_uri',
    type=str,
    default='http//localhost:5000',
    help='Mlflow URI.'
)
```

5. Use mlflow functionality inside your script and check the results by running the script with prescribed number of hidden neurons
```python
def main():
    args = get_args()
    # Set the remote backend uri
    mlflow.set_tracking_uri(args.mlflow_uri)
    # Set the experiemnt name
    mlflow.set_experiment(args.experiment_name)
    # Start the tracking session
    mlflow.start_run()
    # Initialize experiment layout
    rec_uuid =  str(datetime.datetime.now().timestamp()).replace('.', '')
    rec_path = os.path.join('results', rec_uuid)
    n_hidden = int(args.n_hidden)
    # Set the tags
    mlflow.set_tag('Python', platform.python_version())
    mlflow.set_tag('Machine', platform.machine())
    mlflow.set_tag('Node', platform.node())
    mlflow.set_tag('Platform', platform.platform())
    mlflow.set_tag('Record', rec_uuid)
    # Log the parameter
    mlflow.log_param(f'n_hidden', n_hidden)
    letters = utils.get_letters()
    category_lines = utils.read_files(args.data_path, letters)
    categories = utils.get_categories(args.data_path)
    n_letters = len(letters)
    n_categories = len(categories)
    utils.save_categories(categories)
    rnn = model.RNN(n_letters, n_hidden, n_categories)
    # Log the metric
    loss = iterate(rnn, category_lines, categories, letters, args.n_iters)
    # Save the artefacts
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
    # Log the artifacts
    mlflow.pytorch.log_model(rnn, 'model')
    mlflow.log_artifacts(rec_path)
    # Terminate the tracking session
    print(f'The record {rec_uuid} was created') 
    mlflow.end_run()
```

After configuring and finishing the code you can run the project using the following code
```shell
pip install -U neuro-cli neuro-flow neuro-extras
neuro login
neuro-flow build myimage
neuro-flow mkvolumes
neuro-flow upload ALL
neuro-flow run mlflow
neuro-flow run train --param n_hidden 120
```
