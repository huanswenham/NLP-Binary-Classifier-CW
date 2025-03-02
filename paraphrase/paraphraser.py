from transformers import T5ForConditionalGeneration, T5Tokenizer

class Paraphraser:
    def __init__(self, max_length=1000, temperature=0.9, num_paraphrases=2):
        # Initialise T5 model (base)
        self.tokenizer = T5Tokenizer.from_pretrained("hetpandya/t5-small-tapaco")
        self.model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-small-tapaco")
        # Model configurations
        self.max_length = max_length
        self.temperature = temperature
        self.num_paraphrases = num_paraphrases

    # Function to run the T5 model to paraphrase text
    def paraphrase(self, text):
        # Add task prefix
        input_text = "paraphrase: " + text

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # Generate paraphrased output
        model_output = self.model.generate(inputs['input_ids'], max_length=self.max_length, num_beams=2, num_return_sequences=self.num_paraphrases)

        outputs = []
        for output in model_output:
            generated_sent = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if (generated_sent.lower() != text.lower() and generated_sent not in outputs):
                outputs.append(generated_sent)
        return outputs
