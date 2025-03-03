from parrot import Parrot


class ParrotParaphraser:
    def __init__(self):
        self.model = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    # Function to paraphrase a sentence using Parrot pretrained model
    def paraphrase(self, sentence):
        return self.model.augment(input_phrase=sentence)
    