# External library imports
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report

# Local imports
from deepseek_api import DeepSeekApi

# Configurations
BATCH_SIZE = 1
LOG_FILEPATH = '../deepseek_api_logs/deepseek_dev_baseline.log'
SAVE_CSV_FILEPATH = '../data/deepseek_dev_baseline.csv'


# Main process
def main():
    # Get the training dataset
    dev_df = pd.read_csv("../data/dev_set.tsv", delimiter="\t")
    dev_df = pd.DataFrame(dev_df).reset_index(drop=True)

    # Initialise a new empty dataframe
    new_df = pd.DataFrame()

    # Initialise DeepSeek api
    deepseek = DeepSeekApi()

    # Set up logging to a file
    logging.basicConfig(filename=LOG_FILEPATH, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    index = 0
    while index < len(dev_df):
        try:
            logging.info(f"Processing from index {index}.")
            texts = []
            for i in range(index, min(index + BATCH_SIZE, len(dev_df))):
                text = dev_df.iloc[i, 4]
                texts.append(text)
                logging.info(f"Sentence: {text}")

            predictions = deepseek.batch_pcl_classify(texts, min(BATCH_SIZE, len(dev_df) - index))
            logging.info(f"Predictions: {predictions}")
            # Append a new entry to new DataFrame
            new_df = add_predictions_to_new_df(predictions, index, dev_df, new_df)
            # Save the new dataframe
            new_df.to_csv(SAVE_CSV_FILEPATH, index=False, encoding="utf-8")
            # Increment index by batch size
            index += BATCH_SIZE
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            index += BATCH_SIZE

    # Read the new csv file that contains the model's predictions
    preds_df = pd.read_csv(SAVE_CSV_FILEPATH)
    preds_df = pd.DataFrame(preds_df).reset_index(drop=True)
    
    # Get the predictions
    model_preds = preds_df["prediction"]
    # Get the original labels
    labels  = preds_df["label"]
    # Calculate F1 score
    f1_score = calc_f1(model_preds, labels)
    print(f"The F1 score for DeepSeek v3 PCL binary classifaction: {f1_score}")



def add_predictions_to_new_df(predictions, index, old_df, new_df):
    for i in range(index,  min(index + BATCH_SIZE, len(old_df))):
        entry = old_df.iloc[[i], [4]].copy()
        entry.at[entry.index[0], 'prediction'] = predictions[i - index]
        new_df = pd.concat([new_df, entry], ignore_index=True)
    return new_df

def calc_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return classification_report(labels, preds, output_dict=True)["1"]["f1-score"]



if __name__ == "__main__":
    main()