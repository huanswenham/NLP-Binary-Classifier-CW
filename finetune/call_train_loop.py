from train_loop import train_loop, SEEDS
from classifier import UPSAMPLE, DOWNSAMPLE
from pathlib import Path

import argparse

BASE_DIR = "results"
SEED = 42

def main():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    
    parser.add_argument('--filename', type=str, help="Name of the file to store results in")
    parser.add_argument('--embeddings_model', type=str, help="Embeddings model wrapper to use")

    # Parse the arguments
    args = parser.parse_args()

    print("Embeddings model:", args.embeddings_model)
    print("Embeddings wrapper:", args.embeddings_wrapper)
    print("Classifier:", args.classifier)


    # Create subdirectory if needed
    directory = Path(f"{BASE_DIR}/")
    directory.mkdir(parents=False, exist_ok=True)

    filepath = directory / args.filename
    
    # Call the train_loop with parsed arguments, trying range of unfrozen layers and learning rates
    append = False

    for sampling, max_epochs in [(UPSAMPLE, 3), (DOWNSAMPLE, 5), (None, 3)]:
            for unfreeze_layers in range(2, 6):
                for lr in [2e-5, 3e-5, 4e-5]:
                    train_loop(filepath, sampling, SEED, args.embeddings_model, unfreeze_layers, lr, max_epochs)

if __name__ == "__main__":
    main()
