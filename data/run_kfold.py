import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs


partial_train_df = pd.read_csv('train_set.csv')
internal_dev_df = pd.read_csv('internal_validation_set.csv')
val_df = pd.read_csv('dev_set.csv')

# Combine partial train with internal dev to make the train set
train_df = pd.concat([partial_train_df, internal_dev_df], ignore_index=True)

# Replace NaNs with empty strings
train_df['text'].fillna('', inplace=True)
val_df['text'].fillna('', inplace=True)

# Returns the positive class F1-score
def f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return classification_report(labels, preds, output_dict=True)["1"]["f1-score"]

def train_deberta(fold_idx, learning_rate, batch_size, num_epochs, weight_decay, num_layers_unfrozen, log_prefix="log_lr_"):
    # Load dataset
    train_file = f"fold_train_{fold_idx}.csv"
    val_file = f"fold_val_{fold_idx}.csv"

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Upsample the minority class within the training set
    df_majority = train_df[train_df["label"] == 0]
    df_minority = train_df[train_df["label"] == 1]
    target_minority_size = int(len(df_majority) * (2 / 3)) # Ratio 2:3

    duplication_factor = target_minority_size // len(df_minority)
    remainder = target_minority_size % len(df_minority)
    df_minority_upsampled = pd.concat([df_minority] * duplication_factor, ignore_index=True)
    df_minority_upsampled = pd.concat([df_minority_upsampled, df_minority.sample(n=remainder, random_state=42)], ignore_index=True)

    train_df_upsampled = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Fold {fold_idx}: Training set size after upsampling: {len(train_df_upsampled)} ({train_df_upsampled['label'].value_counts()}), Validation set size: {len(val_df)} ({val_df['label'].value_counts()})")

    # Model configuration with hyperparameters
    model_args = {
        "num_train_epochs": num_epochs,
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "output_dir": f"outputs_lr_{learning_rate}_batch_size_{batch_size}_unfreeze_{num_layers_unfrozen}/fold_{fold_idx}",
        "overwrite_output_dir": True,
        "save_best_model": False,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        # "early_stopping_patience": 3,  # Stop if no improvement after 3 eval steps
        "evaluate_during_training": False,
        # "evaluate_during_training_steps": 2000,
        "use_early_stopping": False,
        # "early_stopping_metric": "f1",
        "early_stopping_metric_minimize": False, # Maximise the F1-score
        "classification_report": True,
        "reprocess_input_data": True,
        "save_steps": -1,
        "fp16": False  # Ensure FP16 is disabled
    }

    # Initialize DeBERTa model
    model = ClassificationModel(
        "deberta",  # Model type
        "microsoft/deberta-base",
        num_labels=2,
        args=model_args,
        # use_cuda=torch.cuda.is_available()
    )

    # Unfreeze the last `num_layers_unfrozen` layers + classifier head
    model_layers = list(model.model.deberta.encoder.layer)
    num_total_layers = len(model_layers)
    layers_to_unfreeze = min(num_layers_unfrozen, num_total_layers)

    for name, param in model.model.named_parameters():
        param.requires_grad = False  # Freeze everything first

    for i in range(num_total_layers - layers_to_unfreeze, num_total_layers):
        for param in model_layers[i].parameters():
            param.requires_grad = True  # Unfreeze selected layers

    # Ensure classifier head is always trainable
    for name, param in model.model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    # Train the model
    model.train_model(
        train_df_upsampled[["text", "label"]],
        eval_df=val_df[["text", "label"]],
        f1=f1
    )

    # Evaluate on validation set
    # result, model_outputs, wrong_predictions = model.eval_model(val_df)
    # print("Validation Results:", result)

    texts = val_df['text'].tolist()

    # Get predictions
    preds, raw = model.predict(texts)

    # Generate classification report
    report = classification_report(val_df['label'], preds, digits=4, output_dict=True)
    print("Classification Report:")
    print(classification_report(val_df['label'], preds, digits=4))

    # Log results to a separate file per job
    base_prefix = "./"
    log_file = f"{base_prefix}{log_prefix}{learning_rate}.csv"
    log_data = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "num_layers_unfrozen": num_layers_unfrozen,
        "f1_class_0": report["0"]["f1-score"],
        "f1_class_1": report["1"]["f1-score"],
        "precision_class_0": report["0"]["precision"],
        "precision_class_1": report["1"]["precision"],
        "recall_class_0": report["0"]["recall"],
        "recall_class_1": report["1"]["recall"],
        "overall_f1": report["macro avg"]["f1-score"]
    }

    log_df = pd.DataFrame([log_data])
    log_df.to_csv(log_file, index=False)

    return report["1"]["f1-score"]

learning_rate = 0.00005
num_epochs = 5
weight_decay = 0.0
batch_size = 16
num_layers_unfrozen = 5

def run_loop(learning_rate, batch_size, num_epochs, weight_decay, num_layers_unfrozen):
    f1_scores = []

    for i in range(1, 6):  # Assuming 5 folds
        f1_score = train_deberta(fold_idx=i, learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs, weight_decay=weight_decay, num_layers_unfrozen=num_layers_unfrozen)
        f1_scores.append(f1_score)

    # Print the average F1-score for this combination
    print(f"Average F1-score for lr={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}, weight_decay={weight_decay}, num_layers_unfrozen={num_layers_unfrozen}: {np.mean(f1_scores)}")
    print(f1_scores)

for batch_size in [16, 32]:
    for num_layers_unfrozen in [2, 3, 5]:
        run_loop(learning_rate, batch_size, num_epochs, weight_decay, num_layers_unfrozen)