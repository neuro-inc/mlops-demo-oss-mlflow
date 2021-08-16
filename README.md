# MLFlow Tutorial in Neuro MLOps Platform 

This tutorial shows how to prepare, monitor and test ML pipeline with MLFlow.
The core problem is an adaption of the [NLP FROM SCRATCH: CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) tutorial to the [Neu.ro platform](https://neu.ro).

## Quick Start

Sign up at [app.neu.ro](https://app.neu.ro) and setup your local machine according to [instructions](https://docs.neu.ro/).

Good examples could be found in the [repository](https://github.com/amesar/mlflow-examples)
 

See [Help.md](HELP.md) for the detailed Neuro Project Template Reference.

Install and log in the platform
```shell
pip install -U neuro-cli neuro-flow
neuro login
```

Look at the live.yml and set the defaults
```yaml
defaults:
  preset: cpu-large
  life_span: 1d
```

Look through the main image config
```shell
images:
  myimage:
    ref: image:$[[ flow.project_id ]]:latest
    dockerfile: $[[ flow.workspace ]]/Dockerfile
    context: $[[ flow.workspace ]]/
    build_preset: cpu-large
```

Buuild the main image
```shell
neuro-flow build myimage
```

Look through the volumes in live.yml
```yaml
volumes:
  data:
    remote: storage:$[[ flow.project_id ]]/data
    mount: /project/data
    local: data
  code:
    remote: storage:$[[ flow.project_id ]]/code
    mount: /project/code
    local: code
  config:
    remote: storage:$[[ flow.project_id ]]/config
    mount: /project/config
    local: config
  results:
    remote: storage:$[[ flow.project_id ]]/results
    mount: /project/results
    local: results
  project:
    remote: storage:$[[ flow.project_id ]]
    mount: /project
    local: .
```

Create the volumes in the platform
```shell
neuro-flow mkvolumes
```

Upload the volumes contents
```shell
neuro-flow upload ALL
```

Look through the Postgresql config
```yaml
postgres:
    image: postgres:12.5
    name: $[[ flow.title ]]-postgres
    preset: cpu-small
    http_port: 5432
    http_auth: False
    life_span: 30d
    detach: True
    volumes:
      - disk:mlops-demo-oss-mlflow-postgres:/var/lib/postgresql/data:rw
    env:
      POSTGRES_PASSWORD: password
      POSTGRES_INITDB_ARGS: ""
      PGDATA: /var/lib/postgresql/data/pgdata
```

Create a persistent disk for Postgresql, export it as an env variable
Run a Postgresql server:
```shell
neuro disk create 1G --timeout-unused 30d --name mlops-demo-oss-mlflow-postgres
export MLFLOW_STORAGE=mlops-demo-oss-mlflow-postgres
neuro-flow run postgres
```

Look through the mlflow job config
```yaml
mlflow:
    image: neuromation/mlflow:1.11.0
    name: $[[ flow.title ]]-mlflow-server
    preset: cpu-medium
    http_port: 5000
    http_auth: False
    browse: True
    life_span: 30d
    detach: True
    volumes:
      - storage:${{ flow.flow_id }}/mlruns:/usr/local/share/mlruns
    cmd: |
      server --host 0.0.0.0
        --backend-store-uri=postgresql://postgres:password@${{ inspect_job('postgres').internal_hostname_named }}:5432
        --default-artifact-root=/usr/local/share/mlruns
```

Run an MLFlow server for experiment and model tracking. In this setup, we imply personal use of MLFlow server (each user will connect to their own server), and export its URI to the env variable.

```shell
neuro-flow run mlflow
export MLFLOW_URI=https://demo-oss-names-mlflow-server.jobs.default.org.neu.ro
```

Look at the train job config
```yaml
  train:
    image: $[[ images.myimage.ref ]]
    name: $[[ flow.title ]]-experiment
    pass_config: True
    params:
      mlflow_storage:
        descr: Storage path, where MLFlow server stores trained model binaries
      mlflow_uri:
        descr: MLFlow server URI
      n_hidden:
        descr: Number of neurons in the hidden layer
    volumes:
      - $[[ upload(volumes.data).ref_ro ]]
      - $[[ upload(volumes.code).ref_ro ]]
      - $[[ upload(volumes.config).ref_ro ]]
      - $[[ volumes.results.ref_rw ]]
    env:
      EXPOSE_SSH: "yes"
      PYTHONPATH: /usr/project
      PROJECT: /usr/project
      MLFLOW_STORAGE: ${{ params.mlflow_storage }}
      MLFLOW_URI: ${{ params.mlflow_uri }}
      TRAIN_IMAGE_REF: ${{ images.myimage.ref }}
    bash: |
        python -u $[[ volumes.code.mount ]]/train.py \
          --data_path $[[ volumes.data.mount ]] \
          --n_hidden ${{ params.n_hidden }
```

```python
def main():
    args = get_args()
    
    # 1. Set the remote backend uri
    mlflow.set_tracking_uri(os.environ['MLFLOW_URI'])

    # 2. Setting the experiemnt name
    mlflow.set_experiment(args.experiment_name)
    
    # 3. Starting the tracking session
    mlflow.start_run()
    tmp_dir = tempfile.TemporaryDirectory()

    # 4. Initializing experiment layout
    rec_uuid =  str(datetime.datetime.now().timestamp()).replace('.', '')
    record_path = os.path.join('results', rec_uuid)
    record_path_ = os.path.join(mlflow.get_artifact_uri(), rec_uuid)
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
    loss = perform(rnn, category_lines, categories, letters, args.n_iters, writer)

    # 7. Logging the metric
    mlflow.log_metric('loss', loss)

    # 8. Saving the artefacts
    info = {
        'uuid': rec_uuid,
        'params': {
            'n_hidden': n_hidden
        },
        'metrics': {
            'loss': str(loss)
        },
        'artifacts': [
            'model.pt',
            'chart.png',
            'code.zip'
        ]
    }
    model_path = os.path.join(record_path, 'model.pt')
    chart_path = os.path.join(record_path, 'chart.png')
    code_path = os.path.join(record_path, 'code')
    code_path_ = os.path.join(record_path, 'code.zip')
    cache_path = os.path.join('code', '__pycache__')
    info_path = os.path.join(record_path, 'info.json')

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
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # 9. Logging the artifacts
    mlflow.pytorch.log_model(rnn, 'model')
    mlflow.log_artifacts(record_path)
    
    # 10. Terminating the tracking session
    print(f'The record {rec_uuid} was created') 
    mlflow.end_run()
```

```shell
neuro-flow run train --param mlflow_storage $MLFLOW_STORAGE --param mlflow_uri $MLFLOW_URI --param n_hidden 120
```


Look at the inference server configs
```yaml
  server:
    http_port: 8080
    http_auth: false
    image: $[[ images.myimage.ref ]]
    detach: False
    life_span: 10d
    params:
      rec_uuid:
        descr: Uuid of the record to be inferenced with
    volumes:
      - $[[ upload(volumes.data).ref_ro ]]
      - $[[ upload(volumes.code).ref_ro ]]
      - $[[ upload(volumes.config).ref_ro ]]
      - $[[ volumes.results.ref_rw ]]
    env:
      EXPOSE_SSH: "yes"
      PYTHONPATH: $[[ volumes.code.mount ]]
    bash: |
        python -u $[[ volumes.code.mount ]]/server.py \
          --data_path $[[ volumes.data.mount ]] \
          --dump_dir $[[ volumes.results.mount ]] \
          --rec_uuid ${{ params.rec_uuid }}
```

Run the filebrowser to detect the artifacts
```shell
    neuro-flow run filebrowser
```

Look at the inference server code
```python
args = get_args()
app = Sanic("inference_server")
net = torch.load(os.path.join(args.dump_dir, args.rec_uuid, 'model.pt'))

@app.route('/')
async def index(request):
    input_line = request.json['line'] if request.json else ''
    predictions = predict.perform(net, input_line)
    return json({'predictions': predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True)
```

Run the inference server
```shell
neuro-flow run server --param rec_uuid 000000000 
```

Run the load tests
```shell
    locust -f rnn/test.py --headless -u 10 -r 100 --host https://job-559448e2-2578-458b-83d1-95ba3285c000.jobs.neuro-compute.org.neu.ro

```