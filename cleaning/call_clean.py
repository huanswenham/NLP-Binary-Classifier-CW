import pandas as pd
from cleaner import Cleaner

MAX_NUM_PARAPHRASES = 6

def main():
    # Get the training dataset
    train_df = pd.read_csv("../data/train_set.csv")
    train_df = pd.DataFrame(train_df)

    # Initialise grammar correction model to clean data
    cleaner = Cleaner()

    for index, _ in train_df.iterrows():
        original_text = train_df.at[index, "text"]
        corrected_text = cleaner.correct_text(original_text)
        train_df.iloc[index, train_df.columns.get_loc('text')] = corrected_text
    
    # Save the cleaned dataframe
    train_df.to_csv("../data/train_set_cleaned.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()