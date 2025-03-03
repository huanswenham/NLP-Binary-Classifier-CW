import pandas as pd
from parrot_paraphraser import ParrotParaphraser

MAX_NUM_PARAPHRASES = 6

def main():
    # Get the training dataset
    train_df = pd.read_csv("../data/train_set.csv")
    train_df = pd.DataFrame(train_df)

    # Initialise a new empty dataframe
    new_df = pd.DataFrame(columns=train_df.columns)

    # Initialise Parrot paraphrasing model
    parrot_paraphraser = ParrotParaphraser()

    # Filter and get only the entries that are labeled as PCL
    pcl_df = train_df[train_df['label'] == 1]

    for index, _ in pcl_df.iterrows():
        original_text = pcl_df.at[index, "text"]
        paraphrases = [original_text]
        output = parrot_paraphraser.paraphrase(original_text)
        # Add the original entry to the new dataframe
        new_df = add_text_entry_to_new_df(original_text, index, pcl_df, new_df)
        # If sentence cannot be paraphrased, skip
        if not output: 
            continue
        for paraphrase in output:
            if len(paraphrases) >= MAX_NUM_PARAPHRASES:
                break
            if paraphrase not in paraphrases:
                paraphrases.append(paraphrase)
                # Append a new entry to new DataFrame
                new_df = add_text_entry_to_new_df(paraphrase[0], index, pcl_df, new_df)
    
    # Add all original non pcl entries to the new dataframe
    non_pcl_df = train_df[train_df['label'] == 0]
    new_df = pd.concat([new_df, non_pcl_df], ignore_index=True)
    
    # Save the new dataframe
    new_df.to_csv("../data/train_paraphrase_upsampled.csv", index=False, encoding="utf-8")


def add_text_entry_to_new_df(sentence, index, old_df, new_df):
    entry = old_df.iloc[[index]].copy()
    entry.at[entry.index[0], 'text'] = sentence
    return pd.concat([new_df, entry], ignore_index=True)

if __name__ == "__main__":
    main()