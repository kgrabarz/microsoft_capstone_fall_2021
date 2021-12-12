
# Microsoft Capstone Project AC297r Fall 2021

This repo contains code and materials for our capstone project investigating the tradeoffs between fairness and privacy. 

* 4 Notebooks 

  *  ACS-Income-Results.ipynb: Notebook that contains results of the ACS Income dataset
  *  Adult-Results.ipynb: Notebook that contains results of the Adult dataset
  *  COMPAS-Results.ipynb: Notebook that contains results of the COMPAS dataset
  *  German-Credit-Results.ipynb: Notebook that contains results of German Credit dataset

* 4 Utility Python files (note: they are automatically imported in the notebooks) 
  
  * data_prep.py: helper functions about data preparation 
  * dp_model_metrics.py: helper functions about differential privacy model metrics
  * dp_synthesis.py: helper functions about synthesizing differentially private data
  * classification_metrics.py: helper functions about binary classfication 

* kanon.py: source code of k-anonymous package

## How to install smartnoise-sdk package

1. Git clone the latest smartnoise-sdk package from GitHub
  
   `git clone https://github.com/opendp/smartnoise-sdk.git`
   
2. Go to synth folder
   `cd synth`
   
