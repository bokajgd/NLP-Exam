#-----# Importing packages #-----#

# General packages
import numpy as np
from numpy import testing
import pandas as pd
import argparse
from pathlib import Path
import joblib
import string

# For tokenisation
import nltk
from nltk import word_tokenize
nltk.download('punkt')

# TF-IDF vectoriser
from sklearn.feature_extraction.text import TfidfVectorizer

#-----# Project desctiption #-----#

'''
 Preprocessing and preperation of data:

The purpose of this script is to prepare and preproces the raw textual data and the admission data needed for training and testing the classification model. This proces includes the following steps:

1. Clean and prepare admission data
2. Extract discharge summaries from note data
3. Remove newborn cases and in-hospital deaths
4. Bind note-data to 30-day readmission information
5. Split into train, validation and test set and balance training data by oversampling positive cases
6. Removal of special characters, numbers and de-identified brackets
7. Vectorise all discharge notes:
   7a.  Remove stop-words, most common words and very rare words (benchmarks need to be defined)
   7b. Create set of TF-IDF weighted tokenised discharge notes
8. Output datasets and labels as CSV-files
'''

#-----# Defining main function #-----#

# Defining main function
def main(args):
    
    notes_file = args.nf 

    admissions_file = args.af

    max_features = args.mf

    ngram_range = args.ng


    NotePreprocessing(notes_file=notes_file,
                      admissions_file=admissions_file,
                      max_features=max_features,
                      ngram_range=ngram_range)

#-----# Defining class #-----#

class NotePreprocessing:

    def __init__(self, notes_file, admissions_file, max_features, ngram_range):
        
        # Setting directory of input data 
        self.data_dir = self.setting_data_directory() 

        # Setting directory of output plots
        self.out_dir = self.setting_output_directory() 

        # Loading notes
        if notes_file is None:

            notes = pd.read_csv(self.data_dir  / "NOTEEVENTS.csv", engine='python', encoding='utf-8', error_bad_lines=False)

        else: 
            notes = pd.read_csv(self.data_dir / notes_file, engine='python', encoding='utf-8', error_bad_lines=False)
            
        # Loading general admission data
        if admissions_file is None:

            admissions = pd.read_csv(self.data_dir / "ADMISSIONS.csv", engine='python', encoding='utf-8', error_bad_lines=False)

        else: 
            
            admissions = pd.read_csv(self.data_dir / admissions_file, engine='python', encoding='utf-8', error_bad_lines=False)

        # Loading patient data for demogrpahic variable
        patients = pd.read_csv(self.data_dir  / "PATIENTS.csv", engine='python', encoding='utf-8', error_bad_lines=False)

        # Pre-processing admissions data using preproccessing function
        preprocessed_admissions = self.preprocess_admissions(admissions)

        # Pre-processing notes data
        discharge_sums = self.preprocess_notes(notes)

        # Merge data frames
        merged_data = self.merge_dataframes(preprocessed_admissions, discharge_sums, patients)

        # Splitting data
        train_split, test_split = self.split_data(merged_data, test_frac=0.2)

        # Oversampling positive cases
        train_split_balanced = self.oversample_positive_cases(train_split)

        # Save balanced train and test set as csv file
        pd.DataFrame(test_split).to_csv(self.out_dir / "test_data.csv")

        pd.DataFrame(train_split_balanced).to_csv(self.out_dir / "train_data.csv")

        # TF-IDF vectorise and label extraction
        (train, test), (train_labels, test_labels) = self.tfidf_vectorisation(train_split_balanced, 
                                                                              test_split, 
                                                                              max_features, 
                                                                              ngram_range)


    #-#-# FUNCTION FOR PREPROCESSING ADMISSIONS DATA #-#-#
    def preprocess_admissions(self, admissions):

        # Convert time columns to datetime
        admissions.ADMITTIME = pd.to_datetime(admissions.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

        admissions.DISCHTIME = pd.to_datetime(admissions.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

        admissions.DEATHTIME = pd.to_datetime(admissions.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

        # Sort by subject ID and admission date
        admissions = admissions.sort_values(['SUBJECT_ID','ADMITTIME'])

        admissions = admissions.reset_index(drop = True)

        # Create collumn containing next admission time (if one exists)
        admissions['NEXT_ADMITTIME'] = admissions.groupby('SUBJECT_ID').ADMITTIME.shift(-1)

        # Create collumn containing next admission type 
        admissions['NEXT_ADMISSION_TYPE'] = admissions.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

        # Replace values with NaN or NaT if readmissions are planned (Category = 'Elective') 
        rows = admissions.NEXT_ADMISSION_TYPE == 'ELECTIVE'

        admissions.loc[rows,'NEXT_ADMITTIME'] = pd.NaT

        admissions.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

        # It is important to replace the removed planned admissions with the next unplanned readmission. 
        # Therefore, I will backfill the removed values with the values from the next row that contains data about an unplanned readmission

        # Sort by subject ID and admission date just to make sure the order is correct
        admissions = admissions.sort_values(['SUBJECT_ID','ADMITTIME'])
        
        # Back fill removed values with next row that contains data about an unplanned readmission
        admissions[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = admissions.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

        # Add collumn contain the calculated number of the days until the next admission
        admissions['DAYS_NEXT_ADMIT']= (admissions.NEXT_ADMITTIME - admissions.DISCHTIME).dt.total_seconds()/(24*60*60)

        # Some of these patients are noted as readmitted before being discharged from their first admission, perhaps due to internal transferral.
        # Therefore, I have to remove these negative values

        # Removing rows for which value in DAYS_NEXT_ADMIT is negative
        admissions = admissions.drop(admissions[admissions.DAYS_NEXT_ADMIT < 0].index) 

        # Change data type of DAYS_NEXT_ADMIT to float
        admissions['DAYS_NEXT_ADMIT'] = pd.to_numeric(admissions['DAYS_NEXT_ADMIT'])

        return admissions


    #-#-# FUNCTION FOR PREPROCESSING NOTES #-#-#
    def preprocess_notes(self, notes):

        # Keeping only discharge summaries
        discharge_sums = notes.loc[notes['CATEGORY'] == 'Discharge summary']

        # Filtering out last note per admission as some admissions have multiple discharge summaries 
        discharge_sums = (discharge_sums.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()

        # Remove new lines ('\n') and carriage returns ('\r')
        discharge_sums.TEXT = discharge_sums.TEXT.str.replace('\n',' ')
        
        discharge_sums.TEXT = discharge_sums.TEXT.str.replace('\r',' ')

        return discharge_sums


    #-#-# FUNCTION FOR MERGING DATAFRAMES #-#-#
    def merge_dataframes(self, admissions, discharge_sums, patients):
        
        # Merge
        adm_discharge_sums = pd.merge(admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME']],
                                discharge_sums[['SUBJECT_ID','HADM_ID','TEXT']], 
                                on = ['SUBJECT_ID','HADM_ID'],
                                how = 'left')

        # Filtering out all cases of NEWBORN admissions
        adm_discharge_sums = adm_discharge_sums[adm_discharge_sums.ADMISSION_TYPE != 'NEWBORN']

        # Filtering out admissions resulting in patient deaths 
        adm_discharge_sums = adm_discharge_sums[adm_discharge_sums.DEATHTIME.isnull()]

        # Removing admissions with no discharge note
        adm_discharge_sums = adm_discharge_sums.drop(adm_discharge_sums[adm_discharge_sums.TEXT.isnull()].index) 

        # Adding a column with a binary 30-day readmission label which is need for classification model (0 = no readmission, 1 = unplnaned readmission)
        adm_discharge_sums['OUTPUT_LABEL'] = (adm_discharge_sums.DAYS_NEXT_ADMIT < 30).astype('int')

        # Merging data with patient data to concatenate gender collumn
        adm_discharge_sums_with_gender = pd.merge(adm_discharge_sums[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME','OUTPUT_LABEL','TEXT']],
                                patients[['SUBJECT_ID', 'GENDER']], 
                                on = ['SUBJECT_ID'],
                                how = 'left')


        return adm_discharge_sums_with_gender

    #-#-# SPLITTING DATA #-#-#
    def split_data(self, adm_discharge_sums, test_frac):

        # Shuffling the data randomly
        adm_discharge_sums = adm_discharge_sums.sample(n = len(adm_discharge_sums), random_state = 42) # Setting random state so results can be replicated

        # Reset index
        adm_discharge_sums = adm_discharge_sums.reset_index(drop = True)

        # Save 20% of the data as test data 
        test = adm_discharge_sums.sample(frac=test_frac,random_state=42)

        # Use the rest of the data as training data
        train = adm_discharge_sums.drop(test.index)

        return train, test

    #-#-# OVERSAMPLING POSITIVE INSTANCES TO CREATE BALANCED TRAINING DATASET #-#-#
    def oversample_positive_cases(self, train):

        # Splitting the training data into positive and negative cases
        # Subsetting positive cases
        rows_pos = train.OUTPUT_LABEL == 1

        # Getting dfs with only positive and negative cases respectively
        train_pos = train.loc[rows_pos]
        
        train_neg = train.loc[~rows_pos]

        # Keeping random negative samples equal to 5x number of positive samples
        train_neg = train_neg.sample(n = len(train_pos)*5, random_state = 42,  replace = True)

        # Cocnatenating positive and negative cases and sampling more positive cases until same length as negative cases
        train_balanced = pd.concat([train_neg, train_pos.sample(n = len(train_neg), random_state = 42,  replace = True)], axis = 0)

        # Shuffling the order of training samples 
        train_balanced = train_balanced.sample(n = len(train_balanced), random_state = 42).reset_index(drop = True)

        return train_balanced


    #-#-# CLEANING TEXT AND TF-IDF VECTORISING NOTES #-#-#
    def tfidf_vectorisation(self, train_balanced, test, max_features, ngram_range):

        # Defining TF-IDF vectorizer 
        tfidf_vect = TfidfVectorizer(max_features = max_features, 
                            tokenizer = self._tokenizer_better, 
                            stop_words = 'english',
                            ngram_range = (1,ngram_range), # Include uni-, bi- or trigrams
                            max_df = 0.8)

        # Fit vectorizer to discharge notes
        tfidf_vect.fit(train_balanced.TEXT.values)

        # Transform our notes into numerical matrices
        tfidf_vect_notes = tfidf_vect.transform(train_balanced.TEXT.values)

        tfidf_vect_test_notes = tfidf_vect.transform(test.TEXT.values)

        # Saving training data vocabulary
        vocab_path = self.out_dir / 'vocab' / 'vocabulary.pkl'

        with open(vocab_path, 'wb') as fw:
            joblib.dump(tfidf_vect.vocabulary_, fw)

        # Saving seperate variables containing classification labels
        train_labels = train_balanced.OUTPUT_LABEL


        test_labels = test.OUTPUT_LABEL

        # Saving them as csv files
        train_labels.to_csv(self.out_dir / "train_labels.csv")
        
        test_labels.to_csv(self.out_dir / "test_labels.csv")

        # Turning sparse matrixes into np arrays
        tfidf_vect_notes_array = tfidf_vect_notes.toarray()

        tfidf_vect_notes_test_array = tfidf_vect_test_notes.toarray()

        # Save dense arrays as csv files
        pd.DataFrame(tfidf_vect_notes_array).to_csv(self.out_dir / "tfidf_train_notes.csv")

        pd.DataFrame(tfidf_vect_notes_test_array).to_csv(self.out_dir / "tfidf_test_notes.csv")

        return (tfidf_vect_notes_array, tfidf_vect_notes_test_array), (train_labels, test_labels)


    #-#-# UTILITY FUNCTIONS #-#-#  

    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        # Setting root directory
        root_dir = Path.cwd()  

        # Setting data directory
        data_dir = root_dir / 'data'  

        return data_dir


    # Defining function for setting directory for the output
    def setting_output_directory(self):
        
        # Setting root directory
        root_dir = Path.cwd()  

        # Setting output directory
        out_dir = root_dir / 'output' 

        return out_dir


    # Define a tokenizer function
    def _tokenizer_better(self, text):    

        # Define punctuation list
        punc_list = string.punctuation+'0123456789'

        t = str.maketrans(dict.fromkeys(punc_list, ''))

        # Remove punctuaion
        text = text.lower().translate(t)

        # Tokenise 
        tokens = word_tokenize(text)

        return tokens
    
# Executing main function when script is run from command line
if __name__ == '__main__':

    #Create an argument parser from argparse
    parser = argparse.ArgumentParser(description = "[INFO] Pre-processing discharge summaries",
                                formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument('-nf', 
                        metavar="--notes_file",
                        type=str,
                        help=
                        "[DESCRIPTION] The path for the file containing clinical notes. \n"
                        "[TYPE]        str \n"
                        "[DEFAULT]     NOTEEVENTS.csv \n"
                        "[EXAMPLE]     -ne NOTEEVENTS.csv",
                        required=False)

    parser.add_argument('-af', 
                        metavar="--admission_file",
                        type=str,
                        help=
                        "[DESCRIPTION] The path for the file containing general admissions data \n"
                        "[TYPE]        str \n"
                        "[DEFAULT]     ADMISSIONS.csv \n"
                        "[EXAMPLE]     -ne ADMISSIONS.csv \n",
                        required=False)
    
    parser.add_argument('-mf',
                        metavar="--max_features",
                        type=int,
                        help=
                        "[DESCRIPTION] The number of features to keep in the vectorised notes \n"
                        "[TYPE]        int \n"
                        "[DEFAULT]     30000 \n"
                        "[EXAMPLE]     -mf 30000 \n",
                        required=False,
                        default=30000)

    parser.add_argument('-ng',
                        metavar="--ngram_range",
                        type=int,
                        help=
                        "[DESCRIPTION] Defines the range of ngrams to include (either 2 or 3) \n"
                        "[TYPE]        int \n"
                        "[DEFAULT]     3 \n"
                        "[EXAMPLE]     -ng 3 \n",
                        required=False,
                        default=3)

    main(parser.parse_args())