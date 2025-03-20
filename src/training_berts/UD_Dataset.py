import json
import gzip

import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import regex as re
from collections import defaultdict, Counter

class UD_Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, label_vocab=None, remove_other=False, add_punctuation=False, add_noise=False, replace_url_num = False, lower_case = False, seed = 5550):

        sys.stderr.write(f"\nLoading dataset with:\nRemove other: {remove_other}\nAdd punctuation: {add_punctuation}\nAdd noise: {add_noise}\nReplace urls and numbers: {replace_url_num}\nLower casing: {lower_case}\n")

        self.lower_case = lower_case
        
        self.sentences, self.labels = [], []

        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
        #self.mail_pattern = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
        self.mail_pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

        self.replacement_symbols = {
            "url": '<url>',#'Ħ',
            "num": '<num>',#'Ĳ',
            "mail": '<mail>',#'Ĵ',
        }

        self.n_nums = 0
        self.n_urls = 0
        self.n_mails = 0
        
        #Seed the random function
        random.seed(seed)
        
        self.added_lines = 0
        
        for path in paths:
            with gzip.open(path, "rb") as file:
                lines = file.read().decode("utf-8").split("\n")
        
            for line in lines:
                if line == "":
                    continue
                data = json.loads(line)
    
                if remove_other:
                    if data["languages"][0] == "other":
                        continue

                text = data["text"].strip()

                if lower_case:
                    text = text.lower()
                
                if replace_url_num:
                    text, n_num = self.number_pattern.subn(self.replacement_symbols["num"], text)
                    text, n_url = self.url_pattern.subn(self.replacement_symbols["url"], text)
                    text, n_mail = self.mail_pattern.subn(self.replacement_symbols["mail"], text)

                    self.n_nums += n_num
                    self.n_urls += n_url
                    self.n_mails += n_mail
                    
                
                self.sentences.append(text)
                #self.labels.append(data["languages"][0])
                langs = self._add_lang(data)

                if add_punctuation:
                    if "other" not in langs:
                        if text[-1] == ".":
                            if random.random() < 0.05:
                                #add !, ? or white space at the end at 5% chance
                                new_line = f"{text[-1]}{random.choices(['!', ' ', '?'], k = 1, cum_weights = [3,5,6])[0]}"
                                self.sentences.append(new_line)
                                self.labels.append(langs)
                                
                                self.added_lines += 1
                        #line does not end in period (.)
                        else:
                            if random.random() < 0.05:
                                new_line = f"{text}{random.choices(['.', ' ', '!', '?'], k = 1, cum_weights = [8,10,12,13])[0]}"
                                self.sentences.append(new_line)
                                self.labels.append(langs)
                                
                                self.added_lines += 1
                                
                        #add - to the beginning
                        if text[0] != "-":
                            if random.random() < 0.05:
                                new_line = f"- {text}"
                                self.sentences.append(new_line)
                                self.labels.append(langs)
    
                                self.added_lines += 1

                if add_noise:
                    if random.random() < 0.075:
                        split = text.split()
                        i = random.randint(0, len(split))
                        split.insert(i, self.replacement_symbols["num"])
                        new_line = " ".join(split)

                        self.sentences.append(new_line)
                        self.labels.append(langs)

                        self.added_lines += 1
                        
        sys.stderr.write(f"Loaded dataset with {self.added_lines} augmented lines out of {len(self.sentences)} total lines.\n")
        if replace_url_num:
            sys.stderr.write(f"Total number of {self.replacement_symbols['num']}: {self.n_nums}, {self.replacement_symbols['url']}: {self.n_urls}, {self.replacement_symbols['mail']}: {self.n_mails}\n")
        language_distribution = Counter([l for langs in self.labels for l in langs])
        sys.stderr.write(f"Language distribution: {language_distribution}\n")
                                            
        #create label vocab and mappings between int and label string

        self.label_vocab = label_vocab
        if not self.label_vocab:
            #self.label_vocab = sorted(list(set(self.labels)))
            self.label_vocab = sorted(list(set([l for label in self.labels for l in label])))
        self.label_to_int = {label: i for i, label in enumerate(self.label_vocab)}
        self.int_to_label = {i: label for i, label in enumerate(self.label_vocab)}

    def __getitem__(self, index):
        return self.sentences[index], self.label_to_int[self.labels[index]]

    def __len__(self):
        return len(self.sentences)

    def _add_lang(self, data) -> list[str]:
        #FIX LATER
        #if isinstance(data["languages"], str):
        #    self.labels.append([data["languages"]])
        if len(data["languages"]) == 0:
            langs = ["other"]
            self.labels.append(langs)
        else:
            langs = [label for label in data["languages"]]
            self.labels.append(langs)

        return langs

    def get_ner_entities(self, path):
        entities = defaultdict(dict)
        for folder in path.iterdir():
            for file in folder.iterdir():
                if file.is_file():
                    lang, cat = file.name.split("_")
                    cat = cat.split(".")[0].upper()
                    with open(file, "r", encoding = "utf-8") as in_file:
                        elements = in_file.read().strip().split("\n")
                    entities[lang][cat] = elements


        return entities

    def find_candidates(self, entities, current_lang, label):
        candidates = set()
        for language in entities.keys():
            if language != current_lang:
                candidates.update(entities[language][label])

        return list(candidates)
    
    def extend_ner(self, paths, entities_path):
        added_lines = 0
        entities = self.get_ner_entities(entities_path)
        
        for path in paths:
            with gzip.open(path, "rb") as file:
                lines = file.read().decode("utf-8").strip().split("\n")

            added_lines += len(lines)
            for line in lines:
                if line == "":
                    continue
                data = json.loads(line)
    
                    
                text = data["text"].strip()

                start, end = data["start_char"], data["end_char"]
                langs = data["language"]
                label = data["label"]

                if len(langs) > 1:
                    lang = None
                else:
                    lang = langs
                candidates = self.find_candidates(entities, lang, label)

                augmented_text = text[:start] + random.choice(candidates) + text[end:]
                
                if self.lower_case:
                    augmented_text = augmented_text.lower()

                self.sentences.append(augmented_text)
                self.labels.append([label for label in data["language"]])
                #self._add_lang(data)

        sys.stderr.write(f"Extended dataset with NER-data, with a total of {added_lines} lines\n.")
        language_distribution = Counter([l for langs in self.labels for l in langs])
        sys.stderr.write(f"Updated language distribution: {language_distribution}\n")

                

class UD_Single_Dataset(UD_Dataset):
    def __init__(self, paths, label_vocab=None, remove_other=False, add_punctuation=False, add_noise=False, replace_url_num = False, lower_case = False):
        super().__init__(paths, label_vocab, remove_other, add_punctuation, add_noise, replace_url_num, lower_case)

    def __getitem__(self, index):
        return self.sentences[index], self.label_to_int[self.labels[index][0]]

    def __len__(self):
        return len(self.sentences)


class UD_Multi_Dataset(UD_Dataset):
    def __init__(self, paths, label_vocab=None, remove_other=False, add_punctuation=False, add_noise=False, replace_url_num = False, lower_case = False):
        super().__init__(paths, label_vocab, remove_other, add_punctuation, add_noise, replace_url_num, lower_case)

    def __getitem__(self, index):
        return self.sentences[index], [self.label_to_int[l] for l in self.labels[index]]

    def __len__(self):
        return len(self.sentences)



class CollateFunctor:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __call__(self, batch):
        sentences, labels = zip(*batch)
        inputs = self.tokenizer(list(sentences), return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        inputs['labels'] = torch.tensor(labels)
        return inputs

class CollateFunctorMulti:
    def __init__(self, tokenizer, max_len, label_vocab):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_vocab = label_vocab

    def _label_zeros(self, labels):
        NUM_LABELS = len(self.label_vocab)
        batch_size = len(labels)
        """
        new_labels = torch.zeros(8,5)
        for row in range(len(labels)):
            for col in labels[row]:
                new_labels[row][col] = 1
        """
        new_labels = torch.zeros(batch_size, NUM_LABELS)
        for i in range(len(labels)):
            for l in labels[i]:
                new_labels[i][l] = 1
        
        return new_labels
    
    def __call__(self, batch):
        sentences, labels = zip(*batch)
        inputs = self.tokenizer(list(sentences), return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        inputs['labels'] = self._label_zeros(labels)
        return inputs

if __name__ == "__main__":
    #dataset = UD_Dataset(["../data/test.jsonl.gz"])
    #training_sets = ["data/other.jsonl.gz", "data/openlid_big2.jsonl.gz", 'data/gold_train.jsonl.gz', 'data/big_tatoeba_no_other.jsonl.gz', "data/varied_NLP.jsonl.gz", 'data/silver_train.jsonl.gz']
    #dataset = UD_Single_Dataset(training_sets, add_punctuation=True, add_noise=True, replace_url_num = True)
    training_sets = ["../data/multilabel_tatoeba_sentences_v2.jsonl.gz", "../data/multilabel_ud_sentences_v2.jsonl.gz", "../data/multilabel_bitext_sentences.jsonl.gz"]
    dataset = UD_Multi_Dataset(training_sets, add_punctuation=True, add_noise=True, replace_url_num = True)
