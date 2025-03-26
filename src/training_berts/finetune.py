import os
import argparse
import json
import gzip
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import sys
#from typing import Literal
#from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MultilabelAccuracy, MulticlassF1Score, MulticlassAccuracy
import torchmetrics.classification
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AddedToken

from utils import print_cm, report, multi_report
from UD_Dataset import UD_Multi_Dataset, UD_Single_Dataset, CollateFunctor, CollateFunctorMulti

import wandb

# set TOKENIZERS_PARALLELISM so that it doesn't annoy us
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = {
    "roberta" : "xlm-roberta-base",
    "roberta_large" : "FacebookAI/xlm-roberta-large",
    "scandibert" : "vesteinn/ScandiBERT",
    "norbert" : "ltg/norbert3-base",
    "norbert_large" : "ltg/norbert3-large",
    "norbert_xs" : "ltg/norbert3-xs",
    "norbert_small": "ltg/norbert3-small",
}

DATASETS = {
    "tatoeba": "../data/multilabel_tatoeba_sentences_v2.jsonl.gz",
    "ud": "../data/multilabel_ud_sentences_v2.jsonl.gz",
    "bitexts": "../data/multilabel_bitext_sentences.jsonl.gz",
    "varied_NLP": "../data/varied_NLP_cleaned.jsonl.gz",
}

DATASETS_MODIFIED = {
    "bitexts_no_nb_da" : "../data/multilabel_bitext_sentences_no_nb_da.jsonl.gz",
    "bitexts_reduced_da": "../data/multilabel_bitext_sentences_reduced_danish.jsonl.gz",
    "tatoeba_reduced_other": "../data/multilabel_tatoeba_sentences_v2_reduced_other.jsonl.gz"
}

NER_DATASETS = {
    "ner_ud" : "../ner/training_data/multilabel_ud_ne_filled_agg.jsonl.gz",
    "ner_tatoeba" : "../ner/training_data/multilabel_tatoeba_ne_filled_agg.jsonl.gz",
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str, default='xlm-roberta-base', help='The model to use')
    parser.add_argument("--id", type = str, required = True, help = "unique id")
    parser.add_argument('--model', type=str, choices = MODELS.keys(), required = True, help= f"The model to use, available models: {', '.join(MODELS.keys())}")
    parser.add_argument('--train', type=str, nargs = "+", choices = DATASETS.keys(), default = ["tatoeba", "ud"], help = "Which datasets to use for training")
    parser.add_argument('--train_mod', type=str, nargs = "+", choices = DATASETS_MODIFIED.keys(), default = None, help = "Which modified datasets to use for training")
    parser.add_argument('--ner', type=str, nargs = "+", choices = NER_DATASETS.keys(), default = None, help = "Which NER datasets to use for training")
    parser.add_argument('--all', action = "store_true", help = "Whether or not to use all the data")
    parser.add_argument('--val', type=str, default = "../data/validation_annotated.jsonl.gz", help = "Path to validation set")
    parser.add_argument('--test', type=str, default = "../data/test_other_2.jsonl.gz", help = "Path to test set")
    parser.add_argument('--save', action = "store_true", help = "Whether or not to save the model")
    parser.add_argument('--add_punctuation', action = "store_true", help = "Whether or not to randomly add sentence final punctuation marks 10% chance")
    parser.add_argument('--normalize_url_num', action = "store_true", help = "Whether or not to normalize urls and numbers into <url> and <num>")
    parser.add_argument('--add_noise', action = "store_true", help = "Whether or not to add random <num> tokens into the text")
    parser.add_argument('--lower_case', action = "store_true", help = "Whether or not to lower case all data")
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--epochs', type=int, default=5, help='The number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--warmup_steps', type=float, default=0.02, help='The number of warmup steps given in percent of all training steps')
    parser.add_argument('--gradient_clipping', type=float, default=10.0, help='The gradient clipping value')
    parser.add_argument('--patience', "-p", type=int, default = 3, help = "The number of iterations allowed with decreased validation loss before stopping early")
    parser.add_argument('--num_labels', type = int, default = 5, help = "The number of labels the model considers")
    parser.add_argument('--dropout', type = float, default = 0.1, help = "Amount of dropout")
    parser.add_argument('--scheduler', type=str, default = "cosine", choices = ["linear", "cosine"], help = "Which learning rate scheduler to use")
    parser.add_argument('--problem', type=str, default = "multi", choices = ["single", "multi"], help = "Choose problem type, single label or multi label")
    parser.add_argument('--wandb_project', type=str, default=None, help = "IF wandb project is specified, wandb logging is activated.")
    parser.add_argument('--early_stopping', type=str, default=None, choices = ["metric", "loss"], help = "Criterion for early stopping: 'metric' or 'loss'. If not provided, early stopping is disabled.")
    parser.add_argument("--sub_epoch_val", action = "store_true", help = "Validate model after every 100 steps")
    
    args = parser.parse_args()

    if args.all:
        args.train = [key for key in DATASETS.keys()]
    
    return args


def init(args):
    # set random seed
    torch.manual_seed(args.seed)
    
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    tokenizer_path = MODELS[args.model]
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir="./cache"
    )
    if args.normalize_url_num:
        tokenizer.add_tokens([AddedToken("<url>", lstrip=True), AddedToken("<num>", lstrip=True), AddedToken("<mail>", lstrip=True)])
    
    if args.problem == "single":
        problem_type = 'single_label_classification'
    elif args.problem == "multi":
        problem_type = "multi_label_classification"
    
    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[args.model],
        cache_dir="./cache",
        trust_remote_code=True,
        num_labels=args.num_labels,
        problem_type=problem_type
    ).to(device)

    #resize embedding layer to incorporate the two added tokens
    model.resize_token_embeddings(len(tokenizer))
    
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {number_of_parameters:,}")

    return device, tokenizer, model

def get_datasets(args, tokenizer):     
    training_sets = [DATASETS[dataset] for dataset in args.train]

    if args.train_mod is not None:
        training_sets.extend([DATASETS_MODIFIED[dataset] for dataset in args.train_mod])
    
    sys.stderr.write(f"Loading {training_sets}\n")
    
    if args.problem == "single":
        dataset = UD_Single_Dataset
    elif args.problem == "multi":
        dataset = UD_Multi_Dataset
    
    train_set = dataset(training_sets, add_punctuation = args.add_punctuation, add_noise = args.add_noise, replace_url_num = args.normalize_url_num, lower_case = args.lower_case)

    #add NER
    if args.ner is not None:
        ner_training_sets = [NER_DATASETS[dataset] for dataset in args.ner]
        sys.stderr.write(f"Loading NER datasets:{ner_training_sets}\n")
        #Hardcoded entities path
        train_set.extend_ner(ner_training_sets, Path("../ner/entities"))
    
    val_set = dataset([args.val], label_vocab = train_set.label_vocab, replace_url_num = args.normalize_url_num, lower_case = args.lower_case)
    test_set = dataset([args.test], label_vocab = train_set.label_vocab, replace_url_num = args.normalize_url_num, lower_case = args.lower_case)

    if args.problem == "single":
        collator = CollateFunctor(tokenizer, 512)
    elif args.problem == "multi":
        collator = CollateFunctorMulti(tokenizer, 512, train_set.label_vocab)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=collator
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator
    )

    return train_set, train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, lr_scheduler, device, args, epoch: int):
    model.train()
    #Update every 5%
    update_interval = max(1, int(len(train_loader) * 0.05))
    progress_bar = tqdm(train_loader, desc="Training", miniters = update_interval, dynamic_ncols=True, mininterval=25.0)
    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad()

        # forward pass
        loss = model(**batch).loss

        # backward pass
        loss.backward()

        #Clip gradients, should it be performed by default?
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        
        # update weights
        optimizer.step()
        lr_scheduler.step()
        
        if i % update_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        #wandb logging
        if args.wandb_project is not None:
            if i % 20 == 0:
                
                #Calculate gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                wandb.log({
                    'train/training_loss': loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/gradient_norm': total_norm,
                }, step=epoch * len(train_loader) + i)

def train_sub_epoch(model, train_loader, val_loader, optimizer, lr_scheduler, device, metrics, args, epoch: int):
    model.train()
    #Update every 5%
    update_interval = max(1, int(len(train_loader) * 0.05))
    progress_bar = tqdm(train_loader, desc="Training", miniters = update_interval, dynamic_ncols=True, mininterval=25.0)
    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad()

        # forward pass
        loss = model(**batch).loss

        # backward pass
        loss.backward()

        #Clip gradients, should it be performed by default?
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        
        # update weights
        optimizer.step()
        lr_scheduler.step()
        
        if i % update_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        #wandb logging
        if args.wandb_project is not None:
            if i % 20 == 0:
                
                #Calculate gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                wandb.log({
                    'train/training_loss': loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/gradient_norm': total_norm,
                }, step=epoch * len(train_loader) + i)

        if i % 100 == 0:
            calculated_metrics, val_loss = evaluate(model, val_loader, device, metrics, args)
            for name, metric_value in calculated_metrics.items():
                print(f"{name}: {metric_value.item():.2%}")
            print(f"\nvalidation loss = {val_loss:.4f}")
            
            if args.wandb_project is not None:
                wandb.log({
                    'valid/validation_loss': val_loss,
                    **calculated_metrics
                })

            model.train()

@torch.no_grad()
def evaluate(model, val_loader, device, metrics, args):
    model.eval()
    total_correct, total_samples = 0, 0
    validation_loss = 0.0

    
    update_interval = max(1, int(len(val_loader) * 0.05))
    for batch in tqdm(val_loader, desc="Evaluating", miniters = update_interval):
        batch = batch.to(device)
        outputs = model(**batch)

        loss = outputs.loss

        batch_size = batch['labels'].shape[0]
        #multiply with batch_size to account for varying batch_sizes (the last one)
        validation_loss += loss.item() * batch_size

        preds = torch.sigmoid(outputs.logits)
        targets = batch["labels"]

        for metric in metrics.values():
            metric.update(preds, targets)
        
        total_samples += batch_size

    metric_values = {name: metric.compute() for name, metric in metrics.items()}
    loss = validation_loss / total_samples

    #Reset metrics
    for metric in metrics.values(): metric.reset()
    
    return metric_values, loss


def train(model, args, train_loader, val_loader, int_to_label, label_vocab, device, metrics):
    #Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )

    #Calculate number of warmup steps
    training_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_steps * training_steps)

    print(f"{warmup_steps = } / {training_steps}")

    #Initialize chosen learning rate scheduler
    if args.scheduler == "linear":
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )
    elif args.scheduler == "cosine":
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

    #init variables for early stopping
    best_loss = float("inf")
    best_metric = 0.0
    best_epoch = None
    best_model_weights = None
    patience = args.patience
    
    for epoch in range(args.epochs):
        if args.sub_epoch_val:
            train_sub_epoch(model, train_loader, val_loader, optimizer, lr_scheduler, device, metrics, args, epoch)
        else:
            train_epoch(model, train_loader, optimizer, lr_scheduler, device, args, epoch)
        calculated_metrics, val_loss = evaluate(model, val_loader, device, metrics, args)
        #print(f"Epoch {epoch + 1}: validation accuracy = {accuracy:.2%}, validation loss = {val_loss:.2%}\n")
        print(f"Epoch {epoch+1}\n{'-'*33}")
        for name, metric_value in calculated_metrics.items():
            print(f"{name}: {metric_value.item():.2%}")
        print(f"\nvalidation loss = {val_loss:.4f}")

        # Log validation accuracy and average loss once per epoch
        if args.wandb_project is not None:
            wandb.log({
                'valid/validation_loss': val_loss,
                "epoch" : epoch,
                **calculated_metrics,
                "valid/max_metric" : max(list(calculated_metrics.values())[:5])
            })

        if args.problem == "single":
            report(model, val_loader, int_to_label, label_vocab, device)
        elif args.problem == "multi":
            multi_report(model, val_loader, int_to_label, label_vocab, device)


        print("---------------------------------")
        print(f"Epoch {epoch + 1} finished")
        print("---------------------------------")

        if args.early_stopping == "loss":
            #early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model_weights = deepcopy(model.state_dict())
                #reset patience
                patience = args.patience
            else:
                patience -= 1
                if patience == 0:
                    break
        elif args.early_stopping == "metric":
            #early stopping
            #avg_metric = sum(list(calculated_metrics.values())[1:4]) / len(calculated_metrics)
            max_metric = max(list(calculated_metrics.values())[:5])
            if max_metric > best_metric:
                best_metric = max_metric
                best_epoch = epoch
                best_model_weights = deepcopy(model.state_dict())
                #reset patience
                patience = args.patience
            else:
                patience -= 1
                if patience == 0:
                    break

    if args.early_stopping is not None:
        #load best model at the end
        if args.early_stopping == "loss":
            print(f"The best loss was obatined after {best_epoch+1} epochs, with a loss of: {best_loss}.")
        elif args.early_stopping == "metric":
            print(f"The best metric score was obatined after {best_epoch+1} epochs, with a best metric score of: {best_metric}.")
        print(f"Loading model as it was after epoch {best_epoch+1}.")
        model.load_state_dict(best_model_weights)


def init_metrics(args, device):
    if args.problem == "multi":
        metrics = {
            "valid/Accuracy (threshold 0.1)": MultilabelAccuracy(device=device, threshold=0.1),
            "valid/Accuracy (threshold 0.3)": MultilabelAccuracy(device=device, threshold=0.3),
            "valid/Accuracy (threshold 0.5)": MultilabelAccuracy(device=device),
            "valid/Accuracy (threshold 0.7)": MultilabelAccuracy(device=device, threshold=0.7),
            "valid/Accuracy (threshold 0.9)": MultilabelAccuracy(device=device, threshold=0.9),
            "valid/Hamming Accuracy (threshold 0.7)": MultilabelAccuracy(device=device, threshold=0.7, criteria="hamming"),
            "valid/Hamming Accuracy (threshold 0.5)": MultilabelAccuracy(device=device, threshold=0.5, criteria="hamming"),
            "valid/mla": torchmetrics.classification.MultilabelAccuracy(num_labels=args.num_labels, average='weighted').to(device),
        }
    #metric = MultilabelAccuracy(device=device)
    #metric2 = MultilabelAccuracy(device=device, threshold=0.75)
    elif args.problem == "single":
        metrics = {"valid/Weighted F1": MulticlassF1Score(num_classes=args.num_labels, average="weighted", device=device),
                  "valid/Micro Accuracy": MulticlassAccuracy(num_classes=args.num_labels, average="micro", device=device),
                  "valid/Macro Accuracy": MulticlassAccuracy(num_classes=args.num_labels, average="macro", device=device)}

    return metrics

def init_wandb(args, metrics):
    run = wandb.init(
        # Set the project where this run will be logged
        project=args.wandb_project,
        #Set name of run
        name=f"{args.model}_{args.lr}_{args.batch_size}_{args.id}",
        # Track hyperparameters and run metadata
        config={**vars(args)},
    )
    
    # Define custom validation metrics dynamically
    validation_metrics = ['valid/validation_loss', *metrics.keys()]
    
    wandb.define_metric("epoch")
    for metric in validation_metrics:
        wandb.define_metric(metric, step_metric="epoch")


def main():
    args = parse_arguments()
    print(f"NUMBER OF LABELS : {args.num_labels}")

    device, tokenizer, model = init(args)

    metrics = init_metrics(args, device)
    
    if args.wandb_project is not None:
        init_wandb(args, metrics)

    train_set, train_loader, val_loader, test_loader = get_datasets(args, tokenizer)
    
    train(model, args, train_loader, val_loader, train_set.int_to_label, train_set.label_vocab, device, metrics)

    if args.save:
        #torch.save(model.state_dict(), "/cluster/work/projects/ec30/ec-jonassf/exam")
        path = Path(f"/cluster/work/projects/ec30/ec-jonassf/lid/{args.id}")
        path.mkdir()

        print(f"Saving model to path: {path}")
        
        #save label mapping
        model.config.label2id = train_set.label_to_int
        model.config.id2label = train_set.int_to_label
        
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    if args.wandb_project is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
