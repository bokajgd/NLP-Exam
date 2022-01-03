#-----# Importing packages #-----#
from typing import Text
import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm


#-----# Defining function for argparse max input #-----# 
def max_input(arg):

    arg = int(arg)
    
    # If argument is higher than 250, raise an error 
    if arg > 60 or arg < 0:
        raise argparse.ArgumentTypeError('Value must be an integer between 0 and 250')
    
    return arg

#-----# Project desctiption #-----#

'''
This script contains code for calculating and exporting sentiment scores for larger texts using a pre-trained BERT model finetuned for sentiment analysis. 
Texts are split up into smaller chunks (overlapping) that are suitable for input to the model (max 512 tokens). A sentiment score (1-5) is predicted for each chunk
and an average sentiment score for the entire text is lastly calculated by average chunk scores.
'''

#-----# Defining main function #-----#

# Defining main function
def main(args):

    data = args.d

    chunk_size = args.cs

    overlap = args.ol


    SentimentPrediciton(data = data, 
                        chunk_size = chunk_size,
                        overlap = overlap)

#-----# Defining class #-----#

class SentimentPrediciton:

    def __init__(self, data, chunk_size, overlap):
        
        # Setting directory of input data 
        self.data_dir = self.setting_data_directory() 

        # Setting directory of output plots
        self.out_dir = self.setting_output_directory() 

        # Load data
        df = self.load_data(file_name=data)

        # Remove numbers and special characters
        df['TEXT'] = df['TEXT'].str.replace(r"[\[\]*1234567890-]", ' ')

        # Set variables as self vairables
        self.overlap = overlap

        self.chunk_size = chunk_size        

        # Intialising tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

        # Split text into chunks and add new column
        df['SPLIT_TEXT'] = ''

        for i, text in enumerate(df['TEXT']):
            df['SPLIT_TEXT'][i] = self.split_text(text)

        # Create empty column for overall average setniment scores for each text
        average_sent = []

        # Loop through each text
        for text in tqdm(df['SPLIT_TEXT']):
            
            # Run function for calculating sentiment score
            average_sentiment_score = self.sentiment_score(text)
            
            # Append to list
            average_sent.append(average_sentiment_score)

        # Export list of sentiment scores as csv
        pd.DataFrame(average_sent).to_csv(self.out_dir / f"{data}_sentiment_scores.csv")

    #-#-# FUNCTION FOR SPLITTING TEXT #-#-#
    def split_text(self, text):
        
        # Defining empty lists
        full_split_text = []

        chunk = []

        
        # Calculate number of chunks to split text into
        n_chunks = len(text.split())//(self.chunk_size-self.overlap) + 1

        # For each chunk
        for n in range(n_chunks):
            
            # For the first chunk, take first number of words corresponding to chunk size
            if n == 0:

                chunk = text.split()[:self.chunk_size]
                
                # Append chunk to list 
                full_split_text.append(" ".join(chunk))

            # For all other chunks
            else:
                
                # Extract next chunk with overlap
                chunk = text.split()[n*(self.chunk_size-self.overlap):n*(self.chunk_size-self.overlap) + self.chunk_size]
                
                full_split_text.append(" ".join(chunk))
        
        # Return list containing full text that has been spit into overlapping chunks
        return full_split_text

    #-#-# FUNCTION FOR PREDICTING SENTIMENT SCORES ON A SINGLE TEXT #-#-#
    def sentiment_score(self, split_text, sent_score_list=None):
        
        # If no list to store sentiment scores exist, create one
        if sent_score_list is None:
        
            sent_score_list = []
        
        # Loop through chunks
        for text in split_text:
            
            # Tokenise chunk
            tokens = self.tokenizer.encode(text, return_tensors='pt')

            # Run tokens through model and calculate sentiment score logodds
            result = self.model(tokens)
            
            # Append the transformed sentiment score for the chunk to the list
            sent_score_list.append(int(torch.argmax(result.logits))+1)

            # Take mean of all individual sentiment scores
            mean_sentiment = np.mean(sent_score_list)
        
        return mean_sentiment

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


    # Load data function
    def load_data(self, file_name):

        # Load data
        data = pd.read_csv(self.out_dir / file_name) 

        # Remove first column
        data = data.drop(data.columns[0], axis=1) 

        return data


# Executing main function when script is run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "[INFO] Sentiment Prediction",
                                formatter_class = argparse.RawTextHelpFormatter)

    parser.add_argument('-d', 
                        metavar="--data",
                        type=str,
                        help=
                        "[DESCRIPTION] File containig data. Must have column named 'TEXT'. \n"
                        "[TYPE]        str \n"
                        "[DEFAULT]     test_data.csv \n"
                        "[EXAMPLE]     -d test_data.csv \n",
                        default='test_data.csv',
                        required=False)

    parser.add_argument('-cs', 
                        metavar="--chunk_size",
                        type=int,
                        help=
                        "[DESCRIPTION] Size of chunks in which to partition the text (max 60) \n"
                        "[TYPE]        int \n"
                        "[MAX VALUE]   60 \n"
                        "[DEFAULT]     60 \n"
                        "[EXAMPLE]     -cs 60 \n",
                        default=60,
                        required=False)

    parser.add_argument('-ol', 
                        metavar="--overlap",
                        type=int,
                        help=
                        "[DESCRIPTION] Number of words that overlap between chunk \n"
                        "[TYPE]        int \n"
                        "[DEFAULT]     15 \n"
                        "[EXAMPLE]     -ol 15 \n",
                        default=10,
                        required=False)


    main(parser.parse_args())