# Data Influence

## Usage
How to set up the environment for launching the code?
1. Clone this repository using ```git clone https://github.com/farbverlauf0/DataInfluence.git```
2. Inside DataInfluence folder create your python environment using ```python3 -m venv env```
3. Activate the created environment using ```source ./env/bin/activate``` inside DataInfluence folder
4. Install required python packages using ```pip install -r requirements.txt```


After setting up the environment, you can run the code using ```python run.py --flag_1 value_1 --flag_2 value_2 ... --flag_n value_n```.
Before doing this, check if your python environment is activated.
Available flags: `--data-type`, `--noise-scaler`, `--sampler-type` and `--use-raw-data`.
1. For `--data-type` flag, there is only one available value now: `zillow_prize`
2. For `--sampler-type` flag, there are five available values now: `base`, `random`, `fastif`, `shapley` and `all`
3. For `--noise-scaler` flag, you can specify any non-negative float number. If you don't use this flag or use 0.0 for it then no noise will be added to training data. A higher value adds more noise to the data. Now I recommend to use `--noise-scaler 0.5`


After that, you can find the result metrics in `metrics` folder.
Also you will get graphics for losses inside `graphics` folder if you use `--sampler-type all` mode.

## Creating custom objects
In order to create your own sampler:
1. Create a folder inside `utils/samplers`
2. Implement all your methods inside this folder
3. Import your sampler class inside `__init__.py`

It is also possible to add other datasets and functions for their processing. It will be described later.
