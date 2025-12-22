# Transformer + DRL trading strategy on intraday S&P futures

## **Transformer FlOW**

1. project_feature_eng_model.py --> for feature generation and pickle data set

2. project_transformer_training.py --> for training the transformer based on the pickle file from previous step.

3. project_compressed_generator_transformer.py --> produces a compressed file that stacks together the forecasts of the transformer + the 12 features, the output of this script is fed into the Deep RL

4. project_rl.py --> Trains the Deep RL agents with the compressed generated file

5. project_rl_inference_trans.py --> Is the file for generating the performance of the deep RL in new unseen data, creates performance plots, uses backtesting script inside, gives equity curve, trades taken and training logs.

---

## **LightGBM FLOW**

1. project_feature_eng_model.py --> for feature generation and pickle data set

2. project_lgb_training.py --> trains a LGB with the pickle file generated from previous step

3. project_compressed_generator_lightgbm.py --> produces a compressed file that stacks together the forecasts of the lgb + the 12 features, the output of this script is fed into the Deep RL

4.  project_rl.py --> Trains the Deep RL agents with the compressed generated file

5. project_rl_inference_lgb.py --> Is the file for generating the performance of the deep RL in new unseen data, creates performance plots, uses backtesting script inside, gives equity curve, trades taken and training logs.

---

## Extra files

1. project_backtest.py --> is used by DRL, must be in DIR or it wont work

2. pickle_test.py --> is a testing file to inspect the pickle files generated from other scripts


---
## Dependencies

1) Python version used : Python 3.11.14

2) Ubuntu used : 22.04.5 LTS 

3) Python libraries Used:

Package      ----->         Version
 

aiohappyeyeballs          2.6.1

aiohttp                   3.13.2


aiosignal                 1.4.0

anyio                     4.11.0

arch                      8.0.0

argon2-cffi               25.1.0

argon2-cffi-bindings      25.1.0

arrow                     1.4.0

asttokens                 3.0.0

async-lru                 2.0.5

attrs                     25.4.0

babel                     2.17.0

beautifulsoup4            4.14.2

bleach                    6.3.0

build                     1.3.0

certifi                   2025.11.12

cffi                      2.0.0

charset-normalizer        3.4.4

comm                      0.2.3

contourpy                 1.3.3

cycler                    0.12.1

datasets                  4.4.1

debugpy                   1.8.17

decorator                 5.2.1

defusedxml                0.7.1

dill                      0.4.0

executing                 2.2.1

fastjsonschema            2.21.2

filelock                  3.19.1

fonttools                 4.60.1

fqdn                      1.5.1

frozenlist                1.8.0

fsspec                    2025.9.0

h11                       0.16.0

hf-xet                    1.2.0

httpcore                  1.0.9

httpx                     0.28.1

huggingface-hub           0.36.0

idna                      3.11

ipykernel                 7.1.0

ipython                   9.7.0

ipython_pygments_lexers   1.1.1

ipywidgets                8.1.8

isoduration               20.11.0

jedi                      0.19.2

Jinja2                    3.1.6

joblib                    1.5.2

json5                     0.12.1

jsonpointer               3.0.0

jsonschema                4.25.1

jsonschema-specifications 2025.9.1

jupyter                   1.1.1

jupyter_client            8.6.3

jupyter-console           6.6.3

jupyter_core              5.9.1

jupyter-events            0.12.0

jupyter-lsp               2.3.0

jupyter_server            2.17.0

jupyter_server_terminals  0.5.3

jupyterlab                4.4.10

jupyterlab_pygments       0.3.0

jupyterlab_server         2.28.0

jupyterlab_widgets        3.0.16

kiwisolver                1.4.9

lark                      1.3.1

lightgbm                  4.6.0

MarkupSafe                2.1.5

matplotlib                3.10.7

matplotlib-inline         0.2.1

mistune                   3.1.4

mpmath                    1.3.0

multidict                 6.7.0

multiprocess              0.70.18

nbclient                  0.10.2

nbconvert                 7.16.6

nbformat                  5.10.4

nest-asyncio              1.6.0

networkx                  3.5

notebook                  7.4.7

notebook_shim             0.2.4

numpy                     2.3.4

nvidia-cublas-cu12        12.4.5.8

nvidia-cuda-cupti-cu12    12.4.127

nvidia-cuda-nvrtc-cu12    12.4.127

nvidia-cuda-runtime-cu12  12.4.127

nvidia-cudnn-cu12         9.1.0.70

nvidia-cufft-cu12         11.2.1.3

nvidia-curand-cu12        10.3.5.147

nvidia-cusolver-cu12      11.6.1.9

nvidia-cusparse-cu12      12.3.1.170

nvidia-cusparselt-cu12    0.6.2

nvidia-nccl-cu12          2.21.5

nvidia-nvjitlink-cu12     12.4.127

nvidia-nvtx-cu12          12.4.127

overrides                 7.7.0

packaging                 25.0

pandas                    2.3.3

pandocfilters             1.5.1

parso                     0.8.5

patsy                     1.0.2

pexpect                   4.9.0

pillow                    12.0.0

pip                       25.2

platformdirs              4.5.0

prometheus_client         0.23.1

prompt_toolkit            3.0.52

propcache                 0.4.1

psutil                    7.1.3

ptyprocess                0.7.0

pure_eval                 0.2.3

pyarrow                   22.0.0

pycparser                 2.23

Pygments                  2.19.2

pyparsing                 3.2.5

pyproject_hooks           1.2.0

python-dateutil           2.9.0.post0

python-json-logger        4.0.0

pytz                      2025.2

PyYAML                    6.0.3

pyzmq                     27.1.0

referencing               0.37.0

regex                     2025.11.3

requests                  2.32.5

rfc3339-validator         0.1.4

rfc3986-validator         0.1.1

rfc3987-syntax            1.1.0

rpds-py                   0.28.0

safetensors               0.6.2

scikit-learn              1.7.2

scipy                     1.16.3

Send2Trash                1.8.3

setuptools                80.9.0

six                       1.17.0

sniffio                   1.3.1

soupsieve                 2.8

stack-data                0.6.3

statsmodels               0.14.6

sympy                     1.13.1

TA-Lib                    0.6.8

terminado                 0.18.1

threadpoolctl             3.6.0

tinycss2                  1.4.0

tokenizers                0.22.1

torch                     2.6.0+cu124

tornado                   6.5.2

tqdm                      4.67.1

traitlets                 5.14.3

transformers              4.57.1

triton                    3.2.0

typing_extensions         4.15.0

tzdata                    2025.2

uri-template              1.3.0

urllib3                   2.5.0

wcwidth                   0.2.14

webcolors                 25.10.0

webencodings              0.5.1

websocket-client          1.9.0

wheel                     0.45.1

widgetsnbextension        4.0.15

xxhash                    3.6.0

yarl                      1.22.0

