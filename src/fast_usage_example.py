import fasttext
import torch
from transformers import set_seed
from huggingface_hub import hf_hub_download

from identifiers.mlp import MlpClassifier

if __name__ == '__main__':
    set_seed(42)
    model_path = hf_hub_download(repo_id='ltg/SLIDE-fast',
                                 filename="pytorch_model.bin")
    model = MlpClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    id2label = {i: label for i, label in enumerate(["nb", "nn", "da", "sv", "other"])}
    print("Loading FastText model from HF")
    model_path = hf_hub_download(repo_id='cis-lmu/glotlid',
                                 filename="model.bin")
    print("Loaded FastText model from HF")
    embeddings_model = fasttext.load_model(model_path)
    for text in [
            'En dag i livet', # not predicted nynorsk and danish
            'Jag vill ha deg', # correct
            'Jeg er hvalrossen', # correct
            'Denne fuglen har flydd', # not predicted bokmål
            'not a Scandinavian text at all', # correct
            'i sit berømte værk die normen und ihre bertretung i', # correct
        ]:
        print(text)
        vec = embeddings_model.get_sentence_vector(text)
        with torch.no_grad():
            logits = model(torch.from_numpy(vec))
            greater = logits > 0.5
            indices = greater.squeeze().nonzero()
            if indices.shape[0] == 0:
                indices = torch.Tensor([[4]])
            label = [id2label[idx.item()] for idx in indices]
            print(label)
