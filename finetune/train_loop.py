from classifier import PCLClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

BATCH_SIZE = 16

def train_loop(filepath, sampling, seed, embeddings_model, unfreeze_layers, lr, max_epochs):
    print("=============================================")
    print(f"seed={seed}, sampling={sampling}, embeddings_model={embeddings_model}, unfreeze_layers={unfreeze_layers}, lr={lr}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Load the training set
    train_set = pd.read_csv('data/train_set.csv') # TODO: load based on sampling option
    dev_set = pd.read_csv('data/dev_set.csv')

    # Split the train set into a 90:10 train-validation split
    train_set = train_set.sample(frac=1, random_state=seed)
    split = int(0.9 * len(train_set))
    train_set, val_set = train_set.iloc[:split], train_set.iloc[split:]

    model = PCLClassifier(embeddings_model, unfreeze_layers).to(device)

    best_val_loss = None
    best_model = None
    best_model_epochs = None

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False)

    # Train while intermittently evaluating on the internal validation set
    with open(f'{filepath}.csv', 'w+') as f:
        f.write('sampling,unfreeze_layers,lr,epoch,train_step,avg_train_loss,avg_val_loss,accuracy,class0_recall,class0_precision,class0_f1,class1_recall,class1_precision,class1_f1\n')

        # Training loop over epochs
        for epoch in range(max_epochs):
            model.train()
            
            train_loss, train_batches = 0, 0

            for batch in train_dataloader:
                text = list(batch['text'])
                labels = batch['label'].float().to(device)

                # Forward pass
                outputs = model(text)
                loss = criterion(outputs.squeeze(), labels)
                train_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluate on validation set every 100 training steps
                if train_batches % 100 == 0:

                    model.eval()
                    val_loss, val_batches = 0, 0

                    labels = []
                    preds = []

                    with torch.no_grad():                        
                        for batch in val_dataloader:
                            text = list(batch['text'])
                            batch_labels = batch["label"].float().to(device)
                            batch_outputs = model(text)
                            batch_preds = (batch_outputs > 0.5).long()
                            
                            labels.extend(batch_labels.cpu().numpy())
                            preds.extend(batch_preds.cpu().numpy())

                            loss = criterion(batch_outputs.squeeze(), batch_labels)
                            val_loss += loss.item()
                            val_batches += 1

                    # Compute loss
                    avg_train_loss = train_loss / (train_batches + 1)
                    avg_val_loss = val_loss / val_batches

                    # Compute other metrics and save the result
                    accuracy = accuracy_score(labels, preds)
                    precision_class0, precision_class1 = precision_score(labels, preds, zero_division=0, average=None)
                    recall_class0, recall_class1 = recall_score(labels, preds, zero_division=0, average=None)
                    f1_class0, f1_class1 = f1_score(labels, preds, zero_division=0, average=None)
                    f.write(f'{sampling},{unfreeze_layers},{lr},{epoch},{train_batches},{avg_train_loss},{avg_val_loss},{accuracy}, {recall_class0}, {precision_class0}, {f1_class0}, {recall_class1}, {precision_class1}, {f1_class1}\n')

                    # Save the best model seen so far
                    if best_val_loss is None or avg_val_loss < best_val_loss:
                        best_model = model.state_dict()
                        best_model_epochs = epoch
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), f'{filepath}.pth')

                train_batches += 1

    print(f'Training complete! Running on test set with best model (found at epoch {best_model_epochs})...')
    model.load_state_dict(best_model)
    model.eval()

    all_par_ids = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            text = list(batch["text"])
            labels = batch["label"]
            par_id = batch["par_id"]
            
            # Get predictions
            outputs = model(text).squeeze()
            preds = (outputs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_par_ids.append(par_id)

    # Save the results to a CSV
    results = pd.DataFrame({'par_id': all_par_ids, 'label': all_labels, 'prediction': all_preds})
    results.to_csv(f'{filepath}_results.csv', index=False)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)


    print("accuracy:", accuracy,)
    print("precision:", precision,)
    print("recall:", recall,)
    print("f1_score:", f1)
    print()
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))