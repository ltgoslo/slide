from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fasttext
from pathlib import Path
import numpy as np
import re
import json

from argparse import ArgumentParser

class Norbert:
    def __init__(self, model_path, threshold = 0.9):

        with open(f"{model_path}/config.json", "r", encoding = "utf-8") as cfg_file:
            cfg = json.load(cfg_file)

        self.mapping = cfg["id2label"]
        self.mapping_reverse = cfg["label2id"]

        nllb_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.nllb = fasttext.load_model(nllb_path)

        self.project_dir = Path(__file__).parent

        self.threshold = threshold
        
        # set random seed
        torch.manual_seed(42)

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #### PRINT (REMOVE LATER)!
        print(self.device)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir= f"{self.project_dir}/cache",
            trust_remote_code = True
        )

        self.custom_tokenizer = CustomTokenizer(model_path, f"{self.project_dir}/cache")
        
        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir= f"{self.project_dir}/cache",
            num_labels= len(self.mapping),
            trust_remote_code = True
        ).to(self.device)


    def predict(self, sent):
        prediction, confidence = self.nllb.predict(sent)
        confidence = confidence[0]
        if confidence < self.threshold:
            input_ids = self.tokenizer(sent, return_tensors = "pt", padding = True).to(self.device)
            pred = self.mapping[self.model(**input_ids)["logits"].argmax().item()]
            
            return pred
        else:
            language = prediction[0].replace("__label__", "")
    
            if language.startswith("nob_"):
                return "nb"
            if language.startswith("nno_"):
                return "nn"
            if language.startswith("dan_"):
                return "da"
            if language.startswith("swe_"):
                return "sv"
            if language.startswith("eng_"):
                return "en"
            return "other"

    def batch_predict(self, sents, threshold):
        nllb_preds, confidences = self.nllb.predict(sents)
        flat_confidences = np.array(confidences).flatten()
        
        low_confidences = np.where(flat_confidences < threshold)[0]
        all_indexes = np.arange(len(sents))
        high_confidences_mask = ~np.isin(all_indexes, low_confidences)
        high_confidences = all_indexes[high_confidences_mask]

        to_norbert = []
        for idx in low_confidences:
            to_norbert.append(sents[idx])
        if len(to_norbert) > 0:
            input_ids = self.tokenizer(to_norbert, return_tensors = "pt", padding = True).to(self.device)
            preds = self.model(**input_ids)["logits"].argmax(dim=1)
            preds = [self.mapping[label] for label in preds.tolist()] 

        all_preds = [""]*len(sents)

        #add fasttext
        for i, idx in enumerate(high_confidences):
            language = nllb_preds[idx][0].replace("__label__", "")
            if language.startswith("nob_"):
                language = "nb"
            elif language.startswith("nno_"):
                language = "nn"
            elif language.startswith("dan_"):
                language = "da"
            elif language.startswith("swe_"):
                language = "sv"
            elif language.startswith("eng_"):
                language = "en"
            else:
                language = "other"
            all_preds[idx] = language

        #add norbert
        for i, idx in enumerate(low_confidences):
            language = preds[i]
            all_preds[idx] = language
        

        return all_preds

    def batch_predict_multi(self, sents, threshold, target_lang, sigmoid_threshold = 0.80):
        nllb_preds, confidences = self.nllb.predict(sents)
        flat_confidences = np.array(confidences).flatten()
        
        low_confidences = np.where(flat_confidences < threshold)[0]
        all_indexes = np.arange(len(sents))
        high_confidences_mask = ~np.isin(all_indexes, low_confidences)
        high_confidences = all_indexes[high_confidences_mask]

        target_lang = self.mapping_reverse[target_lang]

        to_norbert = []
        for idx in low_confidences:
            to_norbert.append(sents[idx])
        if len(to_norbert) > 0:
            input_ids = self.tokenizer(to_norbert, return_tensors = "pt", padding = True).to(self.device)

            logits = self.model(**input_ids)["logits"]
            sigmoid_values = torch.sigmoid(logits / 2.0)
            
            target_index_values = sigmoid_values[:, target_lang]
            mask = target_index_values > sigmoid_threshold
            argmax_indices = torch.argmax(logits, dim=1)

            preds = torch.where(mask, torch.tensor(target_lang), argmax_indices)
            
            #preds = self.model(**input_ids)["logits"].argmax(dim=1)
            preds = [self.mapping[label] for label in preds.tolist()]

        all_preds = [""]*len(sents)

        #add fasttext
        for i, idx in enumerate(high_confidences):
            language = nllb_preds[idx][0].replace("__label__", "")
            if language.startswith("nob_"):
                language = "nb"
            elif language.startswith("nno_"):
                language = "nn"
            elif language.startswith("dan_"):
                language = "da"
            elif language.startswith("swe_"):
                language = "sv"
            elif language.startswith("eng_"):
                language = "en"
            else:
                language = "other"
            all_preds[idx] = language

        #add norbert
        for i, idx in enumerate(low_confidences):
            language = preds[i]
            all_preds[idx] = language
        

        return all_preds

    def predict_prob(self, sent, tokenizer, temp = 1):
        input_ids = tokenizer(sent, return_tensors = "pt", padding = True).to(self.device)
        print(tokenizer.decode(input_ids["input_ids"][0]))
        logits = self.model(**input_ids)["logits"]
        sigmoids = torch.sigmoid(logits/temp)[0].tolist()
        softmax = torch.softmax(logits, dim = 1)[0].tolist()

        # Determine the maximum width for each column
        max_widths = [max(len(label), 10) for label in self.mapping.values()]

        # Print the header row
        header = ' '.join(f"{label:<{max_widths[i]}}" for i, label in enumerate(self.mapping.values()))
        print(header)

        formatted_row = ' '.join(f"{value:<{max_widths[i]}.3f}" for i, value in enumerate(logits[0].tolist()))
        print(formatted_row)
        formatted_row = ' '.join(f"{value:<{max_widths[i]}.3f}" for i, value in enumerate(sigmoids))
        print(formatted_row)
        formatted_row = ' '.join(f"{value:<{max_widths[i]}.3%}" for i, value in enumerate(softmax))
        print(formatted_row)

        """
        # Print each row of data, formatted to align with the headers
        for row in zip(logits[0].tolist(), sigmoids, softmax):
            formatted_row = ' '.join(f"{value:<{max_widths[i]}.3%}" for i, value in enumerate(row))
            print(formatted_row)
        
        for label in self.mapping.values():
            print(f"{label:<5}", end = "")
        print()
        for logit in logits[0].tolist():
            #print(f"{logit:<5.3f}\t", end = "")
            formatted_row = ' '.join(f"{value:<{max_widhts[i]}.3f}" for i, value i logits[0].tolist())
        print()
        for s in sigmoids:
            print(f"{s:<5.3f}\t", end = "")
        print()
        for sm in softmax:
            print(f"{sm:<5.3%}\t", end = "")
        print()
        """


class CustomTokenizer:
    def __init__(self, pretrained_tokenizer_path, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path,
            cache_dir= cache_dir,
            trust_remote_code = True
        )

        new_tokens = ['<num>', '<url>', '<mail>']
        #self.tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
        self.tokenizer.add_tokens(new_tokens)

        # Define regex patterns for numbers and URLs
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
        self.mail_pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

        self.replacement_symbols = {
            "url": '<url>',#'Ħ',
            "num": '<num>',#'Ĳ',
            "mail": '<mail>',#'Ĵ',
        }


    def preprocess(self, text):
        text = self.number_pattern.sub(self.replacement_symbols["num"], text)
        text = self.url_pattern.sub(self.replacement_symbols["url"], text)
        text = self.mail_pattern.sub(self.replacement_symbols["mail"], text)

        return text

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying tokenizer to retain all its methods
        return getattr(self.tokenizer, attr)

    def __call__(self, text, **kwargs):
        preprocessed_text = self.preprocess(text)
        return self.tokenizer(preprocessed_text, **kwargs)



def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="935536", help="The model to use for langid")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    model_path = f"/cluster/work/projects/ec30/ec-jonassf/lid/{args.model}"

    nb = Norbert(model_path)
    print("model loaded")
    inp = None

    while inp not in ["q", "quit"]:
        inp = input("Sentence: ")
        nb.predict_prob(inp, tokenizer = nb.custom_tokenizer)


    

