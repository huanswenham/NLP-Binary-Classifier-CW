import pandas as pd
from deepseek_api import DeepSeekApi

MAX_NUM_PARAPHRASES = 6

def main():
    # Get the training dataset
    train_df = pd.read_csv("../data/train_set_cleaned.csv")
    train_df = pd.DataFrame(train_df)

    # Initialise a new empty dataframe
    new_df = pd.DataFrame(columns=train_df.columns)

    # Initialise DeepSeek api
    deepseek = DeepSeekApi()

    # Filter and get only the entries that are labeled as PCL
    pcl_df = train_df[train_df['label'] == 1].reset_index(drop=True)

    for index, _ in pcl_df.iterrows():
        if index == 1: break
        print(f"Processing entry index {index}")
        original_text = pcl_df.at[index, "text"]
        # Add the original entry to the new dataframe
        new_df = add_text_entry_to_new_df(original_text, index, pcl_df, new_df)

        # Get paraphrase
        paraphrases = [original_text]
        for _ in range(MAX_NUM_PARAPHRASES):
          output = deepseek.rephrase(original_text)
          print(output)
          if output not in paraphrases:
              paraphrases.append(output)
              # Append a new entry to new DataFrame
              new_df = add_text_entry_to_new_df(output, index, pcl_df, new_df)
    
    # Add all original non pcl entries to the new dataframe
    non_pcl_df = train_df[train_df['label'] == 0]
    new_df = pd.concat([new_df, non_pcl_df], ignore_index=True)
    
    # Save the new dataframe
    new_df.to_csv("../data/train_set_cleaned_deepseek_upsampled.csv", index=False, encoding="utf-8")


def add_text_entry_to_new_df(sentence, index, old_df, new_df):
    entry = old_df.iloc[[index]].copy()
    entry.at[entry.index[0], 'text'] = sentence
    return pd.concat([new_df, entry], ignore_index=True)

if __name__ == "__main__":
    main()