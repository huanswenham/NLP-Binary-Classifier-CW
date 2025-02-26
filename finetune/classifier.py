from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset
from embeddings_models import BaseInferenceModel

import torch
import torch.nn as nn

BASE_MODEL = "microsoft/deberta-v3-base"
EMBEDDING_DIM = 768

class PCLClassifier(nn.Module):
    def __init__(self, embeddings_model: BaseInferenceModel, unfreeze_layers: int = 5) -> None:
        super(PCLClassifier, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Used to get sentence embeddings for classification
        self.embeddings_model = embeddings_model(BASE_MODEL, self.device)

        # Single linear layer with sigmoid activation for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 1),
            nn.Sigmoid()
        )
        
        # Freeze the embeddings model
        for param in self.embeddings_model.parameters():
            param.requires_grad = False

        # Unfreeze the specified number of layers
        encoder_layers = self.embeddings_model.model.encoder.layer

        for layer in encoder_layers[-1:-unfreeze_layers-1:-1]:
            for param in layer.parameters():
                param.requires_grad = True

        print(f"\nFroze {unfreeze_layers} layers. Model parameters:")
        for name, param in self.embeddings_model.named_parameters():
            print(f"    {name}: {param.requires_grad}")
        

    def forward(self, x):
        embeds = self.embeddings_model.get_sentence_embeds(x).to(self.device)
        return self.classifier(embeds)
