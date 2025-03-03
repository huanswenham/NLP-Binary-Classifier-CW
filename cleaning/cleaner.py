from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Cleaner:
    def __init__(self):
        # Initialise grammar correction model (base)
        self.tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
        self.model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")

    def correct_text(self, text):
        # Prepend grammer prefix to let model know to correct grammar of text
        input_text = "grammar: " + text

        # Encode input text with tokenizer
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Generate cleaned text that has been grammarly corrected
        output_ids = self.model.generate(input_ids, max_length = 200)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)