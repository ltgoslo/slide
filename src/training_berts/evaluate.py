import random
import time
from typing import List
from tqdm import tqdm
import os
import urllib.request
import json
import torch
import torchmetrics
from smart_open import open
import logging


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="nllb", help="The method to use for language identification, supported methods are: random, cld3, nllb, fasttext, openlid, langid, lingua, langdetect")
    parser.add_argument("--dataset", type=str, default="../data/test_other_2.jsonl.gz", help="The dataset to use for evaluation")
    parser.add_argument("--threshold", type=float, default=0.75, help = "sigmoid threshold for multilabel prediction")
    parser.add_argument("--custom_tokenizer", action = "store_true", help = "Use the custom tokenizer")
    parser.add_argument('--lower_case', action = "store_true", help = "Whether or not to lower case all data")
    parser.add_argument("--max_len", type = int, default = None, help = "Max length of dataset")
    parser.add_argument("--log", type=str, default = None, help = "Path to output log")
    parser.add_argument("--full_log", type=str, default = None, help = "Path to full output log")
    parser.add_argument("--haas", action = "store_true", help = "Evaluate haas dataset")
    return parser.parse_args()


class AbstractLanguageIdentifier:
    def __init__(self, args):
        self.args = args
        self.languages = ["nb", "nn", "da", "sv", "other"]

    # This method should return a list of languages that the given text could be written in
    # A language can be either ("nb", "nn", "da", "sv" or "other")
    def identify(self, text: str) -> List[str]:
        raise NotImplementedError
    

class RandomLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)

    def identify(self, text: str) -> List[str]:
        return [random.choice(self.languages)]


class CLD3LanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import cld3  # pip install pycld3
        self.cld3 = cld3

    def identify(self, text: str) -> List[str]:
        prediction = self.cld3.get_language(text)
        language = prediction.language
        if not prediction.is_reliable:
            return ["other"]
        if language == "no":
            return ["nb"]
        if language not in self.languages:
            return ["other"]
        return [language]
        
class NBIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext
        from huggingface_hub import hf_hub_download
        
        # download model and get the model path
        # cache_dir is the path to the folder where the downloaded model will be stored/cached.
        model_name = "nb-nordic-lid.bin"
        model_path = hf_hub_download("NbAiLab/nb-nordic-lid", model_name, cache_dir = "./cache")
        
        # load the model
        self.model = fasttext.load_model(model_path)

    """def identify(self, text: str) -> List[str]:
        initial_predictions, confidences = self.model.predict(text, k = 4)
        predictions = [p.replace("__label__", "") for p,c in zip(initial_predictions, confidences) if c > 0.25]
        #language = prediction[0].replace("__label__", "")

        final_predictions = []

        for p in predictions:
            if p.startswith("nob"):
                final_predictions.append("nb")
            elif p.startswith("nno"):
                final_predictions.append("nn")
            elif p.startswith("dan"):
                final_predictions.append("da")
            elif p.startswith("swe"):
                final_predictions.append("sv")
            else:
                final_predictions.append("other")

        if len(final_predictions) == 0:
            p = initial_predictions[0].replace("__label__", "")
            if p.startswith("nob"):
                return ["nb"]
            if p.startswith("nno"):
                return ["nn"]
            if p.startswith("dan"):
                return ["da"]
            if p.startswith("swe"):
                return ["sv"]
            return ["other"]

        if final_predictions[0] == "other":
            return ["other"]
        else:
            final_predictions = [l for l in final_predictions if l != "other"]
        
        return list(set(final_predictions))"""


    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob"):
            return ["nb"]
        if language.startswith("nno"):
            return ["nn"]
        if language.startswith("dan"):
            return ["da"]
        if language.startswith("swe"):
            return ["sv"]
        return ["other"]


class GlotLID(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext
        from huggingface_hub import hf_hub_download
        
        # download model and get the model path
        # cache_dir is the path to the folder where the downloaded model will be stored/cached.
        model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir="./cache")
        print("model path:", model_path)
        
        # load the model
        self.model = fasttext.load_model(model_path)

    """def identify(self, text: str) -> List[str]:
        initial_predictions, confidences = self.model.predict(text, k = 4)
        predictions = [p.replace("__label__", "") for p,c in zip(initial_predictions, confidences) if c > 0.25]
        #language = prediction[0].replace("__label__", "")

        final_predictions = []

        for p in predictions:
            if p.startswith("nob_"):
                final_predictions.append("nb")
            elif p.startswith("nno_"):
                final_predictions.append("nn")
            elif p.startswith("dan_"):
                final_predictions.append("da")
            elif p.startswith("swe_"):
                final_predictions.append("sv")
            else:
                final_predictions.append("other")

        if len(final_predictions) == 0:
            p = initial_predictions[0].replace("__label__", "")
            if p.startswith("nob_"):
                return ["nb"]
            if p.startswith("nno_"):
                return ["nn"]
            if p.startswith("dan_"):
                return ["da"]
            if p.startswith("swe_"):
                return ["sv"]
            return ["other"]

        if final_predictions[0] == "other":
            return ["other"]
        else:
            final_predictions = [l for l in final_predictions if l != "other"]
        
        return list(set(final_predictions))"""

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob_"):
            return ["nb"]
        if language.startswith("nno_"):
            return ["nn"]
        if language.startswith("dan_"):
            return ["da"]
        if language.startswith("swe_"):
            return ["sv"]
        return ["other"]
    

class NllbLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        print(model_path)
        self.model = fasttext.load_model(model_path)

    def identify(self, text: str) -> List[str]:
        initial_predictions, confidences = self.model.predict(text, k = 4)
        predictions = [p.replace("__label__", "") for p,c in zip(initial_predictions, confidences) if c > 0.50]
        #language = prediction[0].replace("__label__", "")

        final_predictions = []

        for p in predictions:
            if p.startswith("nob_"):
                final_predictions.append("nb")
            elif p.startswith("nno_"):
                final_predictions.append("nn")
            elif p.startswith("dan_"):
                final_predictions.append("da")
            elif p.startswith("swe_"):
                final_predictions.append("sv")
            else:
                final_predictions.append("other")

        if len(final_predictions) == 0:
            p = initial_predictions[0].replace("__label__", "")
            if p.startswith("nob_"):
                return ["nb"]
            if p.startswith("nno_"):
                return ["nn"]
            if p.startswith("dan_"):
                return ["da"]
            if p.startswith("swe_"):
                return ["sv"]
            return ["other"]

        if final_predictions[0] == "other":
            return ["other"]
        else:
            final_predictions = [l for l in final_predictions if l != "other"]
        
        return list(set(final_predictions))

    """
    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob_"):
            return ["nb"]
        if language.startswith("nno_"):
            return ["nn"]
        if language.startswith("dan_"):
            return ["da"]
        if language.startswith("swe_"):
            return ["sv"]
        return ["other"]"""


class FasttextLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext
        
        path = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

        # download the model if it does not exist
        if not os.path.exists("lid.176.bin"):
            urllib.request.urlretrieve(path, "lid.176.bin")
        
        self.model = fasttext.load_model("lid.176.bin")

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language == "no":
            return ["nb"]

        if language in self.languages:
            return [language]
        return ["other"]


class OpenlidLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext

        path = "https://data.statmt.org/lid/lid201-model.bin.gz"

        # download the model if it does not exist
        if not os.path.exists("lid201-model.bin"):
            urllib.request.urlretrieve(path, "lid201-model.bin.gz")
            os.system("gunzip lid201-model.bin.gz")
        
        self.model = fasttext.load_model("lid201-model.bin")
    
    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob_"):
            return ["nb"]
        if language.startswith("nno_"):
            return ["nn"]
        if language.startswith("dan_"):
            return ["da"]
        if language.startswith("swe_"):
            return ["sv"]
        return ["other"]
    

class LangidLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)     
        import langid  # pip install langid
        self.langid = langid

    def identify(self, text: str) -> List[str]:
        
        prediction = self.langid.classify(text)
        language = prediction[0]
        if language in self.languages:
            return [language]
        return ["other"]
    

class LinguaLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import lingua  # pip install lingua-language-detector
        from lingua import LanguageDetectorBuilder

        self.detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

    def identify(self, text: str) -> List[str]:
        language = self.detector.detect_language_of(text)
        language = language.iso_code_639_1.name.lower()
        if language in self.languages:
            return [language]
        return ["other"]


class LangdetectLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        from langdetect import detect
        self.detect = detect

    def identify(self, text: str) -> List[str]:
        language = self.detect(text)
        if language == "no":
            return ["nb"]
        if language in self.languages:
            return [language]
        return ["other"]
    
class Norbert(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.mapping = {0:"da", 1:"nb", 2:"nn", 3:"other", 4:"sv"}

        # set random seed
        torch.manual_seed(42)

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        model_path = "./norbert-base/"

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ltg/norbert3-base",
            cache_dir="./cache",#"ltg/norbert3-base",
            trust_remote_code = True
        )

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir="./cache",
            num_labels= 5,
            trust_remote_code = True
        ).to(self.device)

    def identify(self, text: str) -> List[str]:
        input_ids = self.tokenizer(text, return_tensors = "pt", padding = True).to(self.device)
        pred = self.mapping[self.model(**input_ids)["logits"].argmax().item()]

        return [pred]
    

class CustomMulti(AbstractLanguageIdentifier):
    def __init__(self, args, tokenizer_path, model_path):
        super().__init__(args)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.mapping = {0:"da", 1:"nb", 2:"nn", 3:"other", 4:"sv"}

        # set random seed
        torch.manual_seed(42)

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if args.custom_tokenizer:
            self.tokenizer = CustomTokenizer(tokenizer_path, "../finetune/cache")
        else:
            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir="../finetune/cache",
                trust_remote_code = True
            )

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir="../finetune/cache",
            num_labels= 5,
            trust_remote_code = True
        ).to(self.device)

        self.args = args
        
        self.model.eval()

    @torch.no_grad()
    def identify(self, text: str) -> List[str]:
        #get input ids
        input_ids = self.tokenizer(text, return_tensors = "pt", padding = True).to(self.device)

        #multilabel
        logits = self.model(**input_ids)["logits"]
        sigmoids = torch.sigmoid(logits)
        #if other > 0.50, output only "other"
        if sigmoids[0][3] > 0.5:
            preds = torch.zeros(5).int()
            preds[3] = 1
        else:
            preds = (sigmoids > self.args.threshold).int()
        
        ids = [torch.where(row == 1)[0].tolist() for row in preds][0]
        #if no pred over threshold, output other
        if len(ids) == 0:
            # Argmax
            #preds = [self.mapping[self.model(**input_ids)["logits"].argmax().item()]]
            preds = ["other"]
        else:
            preds = [self.mapping[i] for i in ids]

        if self.args.full_log is not None:
            return preds, logits, sigmoids, self.tokenizer.decode(input_ids["input_ids"][0])
        return preds

    def logits_sigmoids(self, text: str):
        #get input ids
        input_ids = self.tokenizer(text, return_tensors = "pt", padding = True).to(self.device)

        #multilabel
        logits = self.model(**input_ids)["logits"]
        sigmoids = torch.sigmoid(logits)

        return logits, sigmoids

class CustomSingle(AbstractLanguageIdentifier):
    def __init__(self, args, tokenizer_path, model_path):
        super().__init__(args)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.mapping = {0:"da", 1:"nb", 2:"nn", 3:"other", 4:"sv"}

        # set random seed
        torch.manual_seed(42)

        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir="../finetune/cache",
            trust_remote_code = True
        )

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir="../finetune/cache",
            num_labels= 5,
            trust_remote_code = True
        ).to(self.device)

    def identify(self, text: str) -> List[str]:
        input_ids = self.tokenizer(text, return_tensors = "pt", padding = True).to(self.device)
        pred = self.mapping[self.model(**input_ids)["logits"].argmax().item()]

        return [pred]

class Hybrid(CustomSingle):
    def __init__(self, args, tokenizer_path, model_path):
        super().__init__(args, tokenizer_path, model_path)
        self.threshold = args.threshold
        self.nllb = NllbLanguageIdentifier(args)

    def identify(self, text: str):
        prediction, confidence = self.nllb.model.predict(text)
        confidence = confidence[0]
        #if confidence < 0.80:
        if confidence < self.threshold:
            return super().identify(text)
        else:
            language = prediction[0].replace("__label__", "")
    
            if language.startswith("nob_"):
                return ["nb"]
            if language.startswith("nno_"):
                return ["nn"]
            if language.startswith("dan_"):
                return ["da"]
            if language.startswith("swe_"):
                return ["sv"]
            return ["other"]

class HybridMulti(CustomMulti):
    def __init__(self, args, tokenizer_path, model_path):
        super().__init__(args, tokenizer_path, model_path)
        #self.nllb = NllbLanguageIdentifier(args)
        self.fasttext = GlotLID(args)

    def identify(self, text: str):
        prediction, confidence = self.fasttext.model.predict(text)
        confidence = confidence[0]
        if confidence < 0.95:
            return super().identify(text)
        else:
            language = prediction[0].replace("__label__", "")
    
            if language.startswith("nob_"):
                return ["nb"]
            if language.startswith("nno_"):
                return ["nn"]
            if language.startswith("dan_"):
                return ["da"]
            if language.startswith("swe_"):
                return ["sv"]
            return ["other"]

class StudentIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        self.threshold = args.threshold
        clf_model_path = args.model
        if args.model == 'scandinavian-lid/SLIDE-fast':
            """clf_model_path = hf_hub_download(
                repo_id=args.model,
                filename="epoch=0-step=4591.ckpt",
                token=os.getenv('HF'), # create .env and put read HF token there
                # or authenticate to hf via CLI (on Fox)
            )"""
            clf_model_path = "/fp/homes01/u01/ec-jonassf/scandi_langid/code/epoch=0-step=4591.ckpt"

        self.clf = LitClassifier.load_from_checkpoint(
            clf_model_path, use_weights=False,
        )
        print(f"Running on {self.clf.device}")
        self.clf.eval()
        self.clf = self.clf.classifier
        supported_languages = ["nb", "nn", "da", "sv", "other"]
        self.id2label = {}
        model_path = hf_hub_download(repo_id='cis-lmu/glotlid',
                                     filename="model.bin", cache_dir = "/fp/homes01/u01/ec-jonassf/scandi_langid/finetune/cache")
        self.model = fasttext.load_model(model_path)

        for i, label in enumerate(supported_languages):
            self.id2label[i] = label

    def identify(self, text: str) -> List[str]:
        vec = self.model.get_sentence_vector(text)
        with torch.no_grad():
            logits = self.clf(torch.from_numpy(vec))
            greater = logits > self.threshold
            indices = greater.squeeze().nonzero()
            if indices.shape[0] == 0:
                indices = torch.Tensor([[4]])
            label = [self.id2label[idx.item()] for idx in indices]
        return label

    def get_confidence_scores(self, text: str):
        vec = self.model.get_sentence_vector(text)
        with torch.no_grad():
            logits = self.clf(torch.from_numpy(vec))
            greater = logits > self.threshold
            indices = greater.squeeze().nonzero()
            if indices.shape[0] == 0:
                indices = torch.Tensor([[4]])
            label = [self.id2label[idx.item()] for idx in indices]
        return label, torch.softmax(logits)


class Fastbank(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import sys
        sys.path.append("/mnt/d/Master/Utils/bokmaal_nynorsk_langid/")
        from fastbank import FastBank

        self.mapping = {"DK" : "da", "BM" : "nb", "NN" : "nn", "SW" : "sv"}

        self.fastbank = FastBank()

    def identify(self, text: str) -> List[str]:
        preds = self.fastbank.classify(text)
        for i in range(len(preds)):
            try:
                preds[i] = self.mapping[preds[i]]
            except:
                preds[i] = "other"
        return preds
    
class FastSpell(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        from fastspell import FastSpell
        self.fsobj = FastSpell("sv", mode = "aggr")

    def identify(self, text: str) -> List[str]:
        pred = self.fsobj.getlang(text)
        if pred == "no":
            pred = "nb"
        if pred not in self.languages:
            pred = "other"

        return [pred]



class CustomTokenizer:
    def __init__(self, pretrained_tokenizer_path, cache_dir):
        from transformers import AutoTokenizer
        import regex as re
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


def print_confusion_matrix(confusion_matrix, supported_languages):
    print("\t\t" + "\t".join(supported_languages))
    for i, row in enumerate(confusion_matrix):
        print("\t" + supported_languages[i], end="\t")
        for cell in row:
            print(int(cell), end="\t")
        print()


def evaluate_haas(args, identifier):
    true = 0
    total = 0
    supported_languages = list(identifier.model.config.id2label.values())
    print(supported_languages)
    with torch.no_grad():
        if args.log is not None:
            out_log = open(f"../data/args.log", "w", encoding = "utf-8")
        with open(args.dataset,
                  'r') as f:
            for line in tqdm(f):
                line = json.loads(line)
                batch = identifier.tokenizer(line['text'], return_tensors = "pt").to(identifier.device)
                out = identifier.model(**batch).logits
                sm = torch.sigmoid(out)
                greater = sm > 0.5
                greater = greater.squeeze()
                indices = greater.nonzero().squeeze(0)
                if indices.nelement() == 0:
                    indices = [3]
                assert len(line['languages']) == 1
                if supported_languages.index(
                        line['languages'][0]) in indices:
                    true += 1
                else:
                    if args.log is not None:
                        out_log.write(f"{json.dumps(line)}, pred:  {str([supported_languages[i] for i in indices])}\n")
                total += 1

        print(true / total)

        if args.log is not None:
            out_log.close()

# prints the confusion matrix, accuracy, precision, recall and macro f1-score
# calculates the loose scores -- if the correct language is in the list of predicted languages, the prediction is considered correct
def evaluate(args, identifier: AbstractLanguageIdentifier):
    print(f"Evaluation of {args.method} started")
    print(f"Loading dataset from {args.dataset}...")
    samples = [json.loads(line) for line in open(args.dataset)]

    if args.max_len is not None:
        random.shuffle(samples)
        samples = samples[:args.max_len]

    supported_languages = identifier.languages
    language_to_index = {language: i for i, language in enumerate(supported_languages)}

    # loose metrics
    loose_accuracy = torchmetrics.Accuracy("binary")
    loose_per_language_f1 = {language: torchmetrics.F1Score("binary") for language in supported_languages}
    loose_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary") for language in supported_languages}

    # strict metrics
    strict_accuracy = torchmetrics.Accuracy("binary")
    overlap_f1 = 0.0
    strict_per_language_f1 = {language: torchmetrics.F1Score("binary", ) for language in supported_languages}
    strict_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary") for language in supported_languages}

    print("Running inference...")
    start_time = time.time()

    for sample in tqdm(samples):
        text = sample["text"]
        if args.lower_case:
            text = text.lower()
        if sample["languages"] == []:
            sample["gold_languages"] = set(["other"])
        else:
            sample["gold_languages"] = set(sample["languages"])

        if args.full_log is not None:
            predictions, logits, sigmoids, tokens = identifier.identify(text)
            sample["predicted_languages"] = set(predictions)
            sample["logits"] = logits
            sample["sigmoids"] = sigmoids
            sample["tokens"] = tokens
        else:
            sample["predicted_languages"] = set(identifier.identify(text))

    if args.log is not None:
        with open(f"{args.log}/{args.method}_wrong_predictions.txt", "w", encoding = "utf-8") as log:
            for sample in samples:
                if sample["gold_languages"] != sample["predicted_languages"]:
                    log.write(f"{sample}\n")

    if args.full_log is not None:
        with open(f"{args.full_log}/{args.method}_all_predictions.txt", "w", encoding = "utf-8") as log:
            for sample in samples:
                log.write(f"TEXT: {sample['text']}\n")
                log.write(f"TOKENIZED TEXT: {sample['tokens']}\n\n")
                log.write(f"Gold labels: {sample['gold_languages']}\n")
                log.write(f"Predicted labels: {sample['predicted_languages']}\n\n")

                logits = sample['logits']
                sigmoids = sample['sigmoids'][0].tolist()
                softmax = torch.softmax(logits, dim = 1)[0].tolist()
                
                # Determine the maximum width for each column
                max_widths = [max(len(label), 10) for label in identifier.mapping.values()]
        
                # Print the header row
                header = ' '.join(f"{label:<{max_widths[i]}}" for i, label in enumerate(identifier.mapping.values()))
                log.write(f"LABEL:   {header}\n")
        
                formatted_row = ' '.join(f"{value:<{max_widths[i]}.3f}" for i, value in enumerate(logits[0].tolist()))
                log.write(f"LOGITS:  {formatted_row}\n")
                formatted_row = ' '.join(f"{value:<{max_widths[i]}.3f}" for i, value in enumerate(sigmoids))
                log.write(f"SIGMOID: {formatted_row}\n")
                formatted_row = ' '.join(f"{value:<{max_widths[i]}.3%}" for i, value in enumerate(softmax))
                log.write(f"SOFTMAX: {formatted_row}\n")

                log.write(f"{'-'*120}\n")
                

    end_time = time.time()

    print("Calculating metrics...")

    #with open("log.txt", "w", encoding = "utf-8") as out:
    for sample in samples:
        gold_languages = sample["gold_languages"]
        predicted_languages = sample["predicted_languages"]

        # loose metrics
        if predicted_languages.issubset(gold_languages):
            loose_accuracy.update(torch.ones(1), torch.ones(1))

            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    loose_per_language_f1[language].update(torch.ones(1), torch.ones(1))
                    loose_per_language_mcc[language].update(torch.ones(1), torch.ones(1))
                elif language not in predicted_languages and language not in gold_languages:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.zeros(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.zeros(1))
        else:
            loose_accuracy.update(torch.zeros(1), torch.ones(1))

            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    loose_per_language_f1[language].update(torch.ones(1), torch.zeros(1))
                    loose_per_language_mcc[language].update(torch.ones(1), torch.zeros(1))
                elif language in predicted_languages:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.ones(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.ones(1))
                elif language in gold_languages:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.ones(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.ones(1))
                else:
                    loose_per_language_f1[language].update(torch.zeros(1), torch.zeros(1))
                    loose_per_language_mcc[language].update(torch.zeros(1), torch.zeros(1))

        # strict metrics
        if predicted_languages == gold_languages:
            strict_accuracy.update(torch.ones(1), torch.ones(1))

            """if len(gold_languages) > 1:
                text = sample["text"]
                print(text)
                print(f"gold: {gold_languages}")
                print(f"pred: {predicted_languages}")
                print("--------------------------------")"""
        else:
            strict_accuracy.update(torch.zeros(1), torch.ones(1))
            """text = sample["text"]
            out.write(f"{text}\n")
            out.write(f"gold: {gold_languages}\n")
            out.write(f"pred: {predicted_languages}\n")
            logits, sigmoids = identifier.logits_sigmoids(text)
            out.write(f"logits: {logits}\n")
            out.write(f"sigmoids: {sigmoids}\n")
            out.write(f"softmax: {torch.softmax(logits, dim = 1)}\n")
            out.write("--------------------------------\n\n")"""
        
        common_languages = len(predicted_languages.intersection(gold_languages))
        overlap_precision = common_languages / len(predicted_languages)
        overlap_recall = common_languages / len(gold_languages)
        if overlap_precision + overlap_recall > 0:
            overlap_f1 += 2 * overlap_precision * overlap_recall / (overlap_precision + overlap_recall)

        for language in supported_languages:
            if language in predicted_languages and language in gold_languages:
                strict_per_language_f1[language].update(torch.ones(1), torch.ones(1))
                strict_per_language_mcc[language].update(torch.ones(1), torch.ones(1))
            elif language in predicted_languages:
                strict_per_language_f1[language].update(torch.ones(1), torch.zeros(1))
                strict_per_language_mcc[language].update(torch.ones(1), torch.zeros(1))
            elif language in gold_languages:
                strict_per_language_f1[language].update(torch.zeros(1), torch.ones(1))
                strict_per_language_mcc[language].update(torch.zeros(1), torch.ones(1))
            else:
                strict_per_language_f1[language].update(torch.zeros(1), torch.zeros(1))
                strict_per_language_mcc[language].update(torch.zeros(1), torch.zeros(1))


    # pretty print the confusion matrix
    print(f"\n# Results for {args.method}:\n")

    print("## Loose metrics")
    
    l1 = f"{args.method} & F1"
    l2 = f"& MCC"
    
    print(f"\tLoose accuracy: {loose_accuracy.compute().item():.2%}")
    print(f"\tLoose macro F1: {sum([loose_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print(f"\tLoose macro MCC: {sum([loose_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print()
    print("### Per-language metrics")
    for language in supported_languages:
        print(f"\t{language}:")
        print(f"\t\tF1: {loose_per_language_f1[language].compute().item():.2%}")
        print(f"\t\tMCC: {loose_per_language_mcc[language].compute().item():.2%}")

        l1 += f" & {(loose_per_language_f1[language].compute().item())*100:.2f}"
        l2 += f" & {(loose_per_language_mcc[language].compute().item())*100:.2f}"

    l1 += f" & {(sum([loose_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages))*100:.2f} \\\\"
    l2 += f" & {(sum([loose_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages))*100:.2f} \\\\"

    print("\n\n## Strict metrics")
    print(f"\tStrict accuracy: {strict_accuracy.compute().item():.2%}")
    print(f"\tOverlap F1: {overlap_f1 / len(samples):.2%}")
    print(f"\tStrict macro F1: {sum([strict_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print(f"\tStrict macro MCC: {sum([strict_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    print()
    print("### Per-language metrics")
    for language in supported_languages:
        print(f"\t{language}:")
        print(f"\t\tF1: {strict_per_language_f1[language].compute().item():.2%}")
        print(f"\t\tMCC: {strict_per_language_mcc[language].compute().item():.2%}")

    print("\n\n## CPU inference time")
    print(f"\tTotal runtime: {end_time - start_time:.2f} seconds")
    print(f"\tms / sentence: {(end_time - start_time) / len(samples) * 1000:.2f} ms")
    print()

    print(l1)
    print(l2)
    print()



def main():
    random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    args = parse_args()
    

    if args.method == "random":
        identifier = RandomLanguageIdentifier(args)
    elif args.method == "cld3":
        identifier = CLD3LanguageIdentifier(args)
    elif args.method == "nllb":
        identifier = NllbLanguageIdentifier(args)
    elif args.method == "fasttext":
        identifier = FasttextLanguageIdentifier(args)
    elif args.method == "openlid":
        identifier = OpenlidLanguageIdentifier(args)
    elif args.method == "langid":
        identifier = LangidLanguageIdentifier(args)
    elif args.method == "lingua":
        identifier = LinguaLanguageIdentifier(args)
    elif args.method == "langdetect":
        identifier = LangdetectLanguageIdentifier(args)
    elif args.method == "glotlid":
        identifier = GlotLID(args)
    elif args.method == "nb":
        identifier = NBIdentifier(args)



    #NORBERT SMALL
    elif args.method == "944014":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/944014", "/cluster/work/projects/ec30/ec-jonassf/lid/944014")
    #NORBERT XS
    elif args.method == "944742":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/944742", "/cluster/work/projects/ec30/ec-jonassf/lid/944742")
    #NORBERT BASE
    elif args.method == "944002":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/944002", "/cluster/work/projects/ec30/ec-jonassf/lid/944002")
    elif args.method == "944008":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/944008", "/cluster/work/projects/ec30/ec-jonassf/lid/944008")

    elif args.method == "944002_hybrid":
        identifier = HybridMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/944002", "/cluster/work/projects/ec30/ec-jonassf/lid/944002")
    elif args.method == "944742_hybrid":
        identifier = HybridMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/944742", "/cluster/work/projects/ec30/ec-jonassf/lid/944742")

    #NORBERT BASE WITH PUNCT AUG DASH AND HYPHEN EARLY STOP BEST_METRIC
    elif args.method == "1091839":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1091839", "/cluster/work/projects/ec30/ec-jonassf/lid/1091839")
    #NO AUG
    elif args.method == "1021994":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1021994", "/cluster/work/projects/ec30/ec-jonassf/lid/1021994")
    #NORBERT BASE WITH PUNCT AUG HYPHEN EARLY STOP BEST_METRIC
    elif args.method == "1021990":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1021990", "/cluster/work/projects/ec30/ec-jonassf/lid/1021990")
    #ONLY CUSTOM TOKENIZER NO AUG
    elif args.method == "1021997":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1021997", "/cluster/work/projects/ec30/ec-jonassf/lid/1021997")
    #NORBERT BASE WITH PUNCT AUG DASH AND HYPHEN EARLY STOP BEST_MLA
    elif args.method == "1092425":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1092425", "/cluster/work/projects/ec30/ec-jonassf/lid/1092425")
    #ONLY AUG
    elif args.method == "1021992":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1021992", "/cluster/work/projects/ec30/ec-jonassf/lid/1021992") 

    #SCANDIBERT
    elif args.method == "1100035":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1100035", "/cluster/work/projects/ec30/ec-jonassf/lid/1100035")
   
    #Norbert base
    elif args.method == "1100063":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1100063", "/cluster/work/projects/ec30/ec-jonassf/lid/1100063")

    #Norbert base
    elif args.method == "1091839":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1091839", "/cluster/work/projects/ec30/ec-jonassf/lid/1091839")

    #DSTILBERT
    elif args.method == "1100033":
        identifier = CustomMulti(args, "/cluster/work/projects/ec30/ec-jonassf/lid/1100033", "/cluster/work/projects/ec30/ec-jonassf/lid/1100033")

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    if args.haas:
        evaluate_haas(args, identifier)
    else:
        evaluate(args, identifier)

    """samples = [json.loads(line) for line in open(args.dataset)]
    for sample in samples[:100]:
        text = sample["text"]
        print(text)
        print(f"gold: {sample['languages']}")
        print(f"pred: {identifier.identify(text)}")
        print("--------------------------------")"""


if __name__ == "__main__":
    main()
