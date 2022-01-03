  <h3 align="center">Investigating Gender Bias in Clinical Discharge Notes</h3>

  <p align="center">
    Exam Project for NLP - Cognitive Science, AU
    <br />
  <p align="center">
    Jakob Grøhn Damgaard, January 2022 <br/>

## Overview
[Scope](#scope) | [Data](#data) | [Usage](#usage) | [Structure](#Structure) | [Contact](#contact)

## Scope
With this project, I aim to investigate the potential presence of human-induced systematic biases in free-text clinical notes and assess the unsought impacts such biases may transmit to predictive models for clinical outcomes. More specifically, I wish to employ sentiment analysis to unravel linguistic differences in narrative discharge notes across genders. To gauge the impact of a potential difference herein, I subsequently examine the association 
between sentiment scores and risk of unplanned readmission. Lastly, I check for differences in sentiment scores across patients that are falsely predicted to not be readmitted within 30 days of discharge (false negatives) and patients that are correctly predicted to be readmitted (true positives) by a predictive model trained on discharge notes. Given the shortage of reliable research on this specific topic, this project should best be regarded as exploratory. Hence, I will purposefully not formulate concrete a priori hypotheses or expectations that may prejudice my discussion but instead let the data analyses speak for themselves. <br> 

## Data
Electronic health records were obtained from the MIMIC-III database (Johnson et al., 2016) and the required data files were retrieved from PhysioNet (Goldberger et al., 2000). The MIMIC-III database is a freely accessible health-data resource which contains an extensive range of both structured and unstructured de-identified data from intensive care admissions of more than 40.000 patients at the Beth Israel Deaconess Medical Center between 2001 and 2012. It comprises detailed information on demographics, test results, free-text clinical notes written by caregivers and ICD-9 procedure and diagnosis codes among others. The clinical notes dataset, which constitutes the foundation of the model exploration in this paper, comprises 2,083,180 human-written notes. The final cleaned discharge notes data used for training, validation and testing comrpise 43,876 notes.<br> 
<br>
Due to the high sensitivity of the data, the MIMIC-III database is, unfortunately, restricted and access to the data requires authorisation. Therefore, I am unable to share the data publicly in the folder and the code is, thus, not runnable on your machine. Please read through the code using the commenting as assistance for understanding the steps taken to complete the preprocessing and the supervised logistic regression classification model. Everyone is free to apply for access to the data from PhysioNet but the process is lengthy:<br>
https://mimic.mit.edu/iii/gettingstarted/


## Usage
**If one has access to the correct data in the data folder, executing the files could be carried out like this:**
<br>
<br>
This section provides a detailed guide for locally downloading the code from GitHub, initialising a virtual Python environment, and installing the necessary requirements Please note, a local installation of Python 3.8.5 or higher and R 4.0.2 is necessary to run the scripts.
To locally download the code, please open a terminal window, redirect the directory to the desired location on your machine and clone the repository using the following command:

 ```bash
git clone https://github.com/bokajgd/Language-Analytics-Exam
 ```
 <br>
Redirect to the directory to which you cloned the repository and then proceed to execute the Bash script provided in the repository for initialising a suitable virtual environment: <br>

 ```bash
 ./create_venv.sh
 ```
 <br>
This command may take a few minutes to finalise since multiple packages and libraries must be collected and updated. Now the code is downloaded locally and all requirements installed, everything can be executed. Before executing code, remember to activate the virtual environment.

Firstly, one should run the following command to get an understanding of how the data pre-processing script is executed and which arguments should be provided:

```bash
# Add -h to view how which arguments should be passed  
python3 src/data_preprocessing.py -h    
usage: data_preprocessing.py [-h] [-nf --notes_file] [-af --admission_file]
                             [-mf --max_features] [-ng --ngram_range]

[INFO] Pre-processing discharge summaries

optional arguments:
  -h, --help            show this help message and exit
  -nf --notes_file      [DESCRIPTION] The path for the file containing clinical notes. 
                        [TYPE]        str 
                        [DEFAULT]     NOTEEVENTS.csv 
                        [EXAMPLE]     -ne NOTEEVENTS.csv
  -af --admission_file  [DESCRIPTION] The path for the file containing general admissions data 
                        [TYPE]        str 
                        [DEFAULT]     ADMISSIONS.csv 
                        [EXAMPLE]     -ne ADMISSIONS.csv 
  -mf --max_features    [DESCRIPTION] The number of features to keep in the vectorised notes 
                        [TYPE]        int 
                        [DEFAULT]     30000 
                        [EXAMPLE]     -mf 30000 
  -ng --ngram_range     [DESCRIPTION] Defines the range of ngrams to include (either 2 or 3) 
                        [TYPE]        int 
                        [DEFAULT]     3 
                        [EXAMPLE]     -ng 3 

```
<br>
By letting the script use the default inputs, it can be executed like this:

```bash

python3 src/data_preprocessing.py

```

This script pre.processes the data and outputs both full data frames for the training and test partion and  TF_IDF vectorised training and test data along with classification labels (vector of 0 or 1s) and the vocabulary obtained when fitting the vectorised.  <br>
<br>
This the latter of the output files from the previous script allows one to run the *readmission_prediction.py* which trains and tests a logistic regression classifier. First, let us examine which arguments the script takes: 

```bash
# Add -h to view how which arguments should be passed  
python3 src/readmission_prediction.py -h
usage: readmission_prediction.py [-h] [-tr --train_data] [-te --test_data]
                                 [-trl --train_labels] [-tel --test_labels]

[INFO] Readmission Prediction

optional arguments:
  -h, --help           show this help message and exit
  -tr --train_data     [DESCRIPTION] The file name of the training data csv 
                       [TYPE]        str 
                       [DEFAULT]     tfidf_train_notes.csv 
                       [EXAMPLE]     -tr tfidf_train_notes.csv 
  -te --test_data      [DESCRIPTION] The file name of the test data csv 
                       [TYPE]        str 
                       [DEFAULT]     tfidf_test_notes.csv 
                       [EXAMPLE]     -tr tfidf_test_notes.csv 
  -trl --train_labels  [DESCRIPTION] The file name of the training labels csv
                       [TYPE]        str 
                       [DEFAULT]     train_labels.csv 
                       [EXAMPLE]     -tr train_labels.csv 
  -tel --test_labels   [DESCRIPTION] The file name of the test labels csv
                       [TYPE]        str 
                       [DEFAULT]     test_labels.csv 
                       [EXAMPLE]     -tr test_labels.csv 
                       
```
Readmission prediction can now be performed using the following command:

```bash

python3 src/readmission_prediction.py

```
This outputs a .csv file containing predicted labels for the test set and print an evaluation table containing model performance metrics in the terminal. This shows the overall perfromance of the logistic regression classifier for prediction 30-day unplanned readmissions. 
<br> 
Upon obtaining the true performance 

## Structure
The structure of the downloaded folder (given data is there) can be viewed using the following command:

```bash
tree -L 2
```

This should yield the following graph:

```bash
.
├── README.md
├── create_venv.sh
├── data
│   ├── ADMISSIONS.csv
│   ├── NOTEEVENTS.csv
│   └── PATIENTS.csv
├── output
│   ├── balanced_test_data.csv
│   ├── balanced_test_data_sentiment_scores.csv
│   ├── balanced_test_labels.csv
│   ├── balanced_tfidf_test_notes.csv
│   ├── lr_predictions_balanced.csv
│   ├── lr_predictions_test_data.csv
│   ├── test_data.csv
│   ├── test_labels.csv
│   ├── tfidf_test_notes.csv
│  ├── tfidf_train_notes.csv
│   ├── tfidf_valid_notes.csv
│   ├── train_data.csv
│   ├── train_labels.csv
│   ├── valid_labels.csv
│   └── vocab
├── requirements.txt
├── src
│   ├── data_preprocessing.py
│   ├── exploratory.ipynb
│   ├── readmission_prediction.py
│   ├── sentiment_scoring.py
│   ├── subset_of_test_data.Rmd
│   └── tests_and_plots.Rmd
└── viz
    ├── ROC-AUC.png
    ├── lr_hyper_params_metrics.png
    ├── model_results.png
    └── sentiment_analysis.png

```

The following table explains the directory structure in more detail:
<br>

| Column | Description|
|--------|:-----------|
```data```| A folder containing the raw data that can be passed as input arguments to the preprocessing script: <br> •	ADMISSIONS.csv: This file contains all information and metadata on admissions <br> •	NOTEEVENTS.csv: This file contains all clinical notes written during all admissions <br>
```src``` | A folder containing the source code (*readmission_prediction.p*y and *data_preprocessing.py*) created to solve the assignment. 
```output``` | An output folder in which the generated data frames containing vectorised notes, labels and the vocabulary used for matching influential features with their tokens <br> •	*.csv* files: Vectorised notes and labels for test, train and validation splits <br> •	*vocab*: This subfolder holds the vocabulary *vocabulary.pkl*
```viz``` | An output folder for the generated visualisations and other visualisations for the README.md file <br> •	*ROC-AUC.png*: Image of AUC-ROC curve <br> •	*most_important.png*: Image showing the most influential features and the weigths <br> •	*preprocessing.png*: Flowchart of the preprocessing pipeline
<br>
Minor disclaimer: A few code chunks in the data_preprocessing.py and readmission_prediction.py have been transferred and modified from scripts used for my bachelor’s project on Learning Latent Feature Representations of Clinical Notes for use in Predictive Models. In general, however, all scripts have been modified and are coded specifically for this exam.
<br>

# Contact
**Jakob Grøhn Damgaard** 
<br />
[bokajgd@gmail.com](mailto:bokajgd@gmail.com?subject=[GitHub]%20stedsans)


[<img align="left" alt="Jakob Grøhn Damgaard | Twitter" width="30px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />][twitter2]
[<img align="left" alt="Jakob Grøhn Damgaard | LinkedIn" width="30px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />][linkedin2]

<br />

</details>

[twitter2]: https://twitter.com/JakobGroehn
[linkedin2]: https://www.linkedin.com/in/jakob-gr%C3%B8hn-damgaard-04ba51144/


 <br>
 
# License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

