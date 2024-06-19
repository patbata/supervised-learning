------------------------------------------------------------------------
Investigating Supervised Learning Model Performance on Wine and Student Performance
------------------------------------------------------------------------
This repository contains the code used to investigate the performance of various
supervised learning models on the Wine and Student Performance datasets. The 
code is written in Python and uses the scikit-learn library for the machine learning models.

All code and documentation is written by Patricia Bata while the data used was
obtained from the UCI database.

------------------------------------------------------------------------
Running the Code
------------------------------------------------------------------------
To run the code, you need to have Python 3.7 or higher installed on your
machine. You can install the required packages by running the following
command in the terminal:
```
pip install -r requirements.txt
```
After installing the required packages, you can run the models by running
the following command in the terminal:
```
python main.py --data {dataset} --model {model_name} --tune {tuning_type}
```
where:
- `{dataset}` is the name of the dataset to use (either `student` or `wine`)
- `{model_name}` is the name of the model to use (either `dt`, `nn`, `boost`, `svm`, `knn`, or `all`)
- `{tuning_type}` is the type of tuning to use (either `preliminary` ,`initial`, `gridsearch`, or `final`)
    - `preliminary` obtains dataset statistics and plots
    - `initial` runs the model with learning (params) and validation curves (hyperparams)
    - `gridsearch` runs the model with the grid search tuning parameters (hyperparams)
    - `final` runs the model with the final tuning parameters (params)
  
For example, to run the decision tree model on the student dataset with
initial tuning, you can run the following command:
```
python main.py --data student --model dt --tune initial
```

To change the configuration of the models, you can modify the configuration
files in the `configs/` directory. The configuration files are in the YAML
format and contain the parameters for the models. The configuration files
are named after the dataset and are located in the `configs/` directory.

------------------------------------------------------------------------
Directory Structure
------------------------------------------------------------------------
The directory structure of the repository is as follows:
```
unsupervised-learning
|--configs/                     - configuration files for the models
    |--student.yaml
    |--wine.yaml
|--src/                         - source code for the project
    |--utils                    - dataset cleaning and parsing functions
        |--DataSetup.py
        |--parsing_utils.py
    |--models                   - machine learning model helper functions
        |--modeling.py
  |--data/                      - dataset files (* denotes the files used)
      |--wine-data/
          |--wine.data*
          |--wine.names
      |--student-data/
          |--student.csv*
          |--student-por.csv
          |--student-merge.R
|--results/                     - results of the models
|--results-analysis.pdf         - written report on the analysis of results
|--main.py                      - main file to run the models
|--eda.ipynb                    - exploratory data analysis notebook
|--requirements.txt             - required packages for the project
|--README.md
```
------------------------------------------------------------------------
References
------------------------------------------------------------------------
- Wine Data Set: https://archive.ics.uci.edu/ml/datasets/wine
- Student Performance Data Set: https://archive.ics.uci.edu/ml/datasets/student+performance
