
# AC297r: Microsoft Capstone Project, Fall 2021

This repo contains code and materials for our capstone project investigating the tradeoffs between fairness and differentially private (DP) synthetic data. We evaluated synthesizers implemented in the [SmartNoise SDK](https://github.com/opendp/smartnoise-sdk) on the Adult, ACS Income, COMPAS, and German Credit data sets. 

* Notebooks: 

  *  `ACS-Income-Results.ipynb`: Notebook containing results on the ACS Income dataset
  *  `Adult-Results.ipynb`: Notebook containing results on the Adult dataset
  *  `COMPAS-Results.ipynb`: Notebook containing results on the COMPAS dataset
  *  `German-Credit-Results.ipynb`: Notebook containing results on the German Credit dataset

* Python files (note: functions are automatically imported in the notebooks): 
  
  * `data_prep.py`: data preparation functions  
  * `dp_model_metrics.py`: functions to calculate fairness and accuracy metrics for DP classification models
  * `dp_synthesis.py`: functions used to train synthesizers and generate DP synthetic data
  * `classification_metrics.py`: functions to calculate fairness and accuracy of classification models trained on DP synthetic data 
  * `preprocessing.py`: multi-label undersampling method to pre-process synthesizer training data and mitigate class imbalance
  * `kanon.py`: implementation of k-anonymity, credit to [Nuclearstar](https://github.com/Nuclearstar/K-Anonymity).

## How to install `smartnoise-sdk` package:

1. Git clone the latest smartnoise-sdk package from GitHub
  
   `git clone https://github.com/opendp/smartnoise-sdk.git`
   
2. Navigate to synth folder

   `cd synth`

3. Install the package by running `setup.py`

   `python setup.py build`
    
   `python setup.py install`
