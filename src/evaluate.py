import logging
import random
import time
import os
import json

import matplotlib.pyplot as plt
from seaborn import heatmap
import torch
import torchmetrics
from smart_open import open
from tqdm import tqdm

from identifiers import (
    AbstractLanguageIdentifier,
    GPT2Identifier,
    RandomLanguageIdentifier,
    FasttextHfHubIdentifier,
    FasttextLanguageIdentifier,
    BERTIdentifier,
    MlpIdentifier,
    FasttextEnsembleIdentifier,
)

OUT_DIR = 'eval_logs/'
GOLD_LANGUAGES = 'gold_languages'
PREDICTED_LANGUAGES = 'predicted_languages'


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="fasttext_hf_hub",
        help="The method to use for language identification",
        choices=("fasttext_ensemble", "bert", "random", "fasttext_hf_hub", "gpt2", "fasttext", "mlp"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../test_data/test_other_2_new.jsonl",
        help="The dataset to use for evaluation",
    )
    parser.add_argument(
        "--model",
        default="cis-lmu/glotlid",
    )
    parser.add_argument(
        "--second_model",
        default="../../OpenLID-v2/new_data/oci_no_pilar_frp/train/openlid-v3.bin",
        help="Used only if method is fasttext_ensemble"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--other_if_below_threshold", action="store_true", help="for bert")
    parser.add_argument("--k", type=int, default=1, help="FastText k")
    parser.add_argument("--languages", default="nb,nn,da,sv,other")
    parser.add_argument("--run_name", default="")
    return parser.parse_args()


def draw_confusion_matrix(confusion_matrix, supported_languages, predicted_languages, args):
    heatmap(
        confusion_matrix,
        annot=True,
        fmt=".0f",
        xticklabels=predicted_languages, yticklabels=supported_languages,
    )
    plt.tight_layout()
    plt.savefig(args.log_fn + '_cm.pdf', format='pdf')


def count_loose(
        predicted_languages,
        gold_languages,
        loose_accuracy,
        supported_languages,
        loose_per_language_f1,
        loose_per_language_mcc,
):
    # loose metrics
    if predicted_languages.issubset(gold_languages):
        loose_accuracy.update(torch.ones(1), torch.ones(1))

        for language in supported_languages:
            if language in predicted_languages and language in gold_languages:
                loose_per_language_f1[language].update(torch.ones(1),
                                                       torch.ones(1))
                loose_per_language_mcc[language].update(torch.ones(1),
                                                        torch.ones(1))
            elif language not in predicted_languages and language not in gold_languages:
                loose_per_language_f1[language].update(torch.zeros(1),
                                                       torch.zeros(1))
                loose_per_language_mcc[language].update(torch.zeros(1),
                                                        torch.zeros(1))
    else:
        loose_accuracy.update(torch.zeros(1), torch.ones(1))

        for language in supported_languages:
            if language in predicted_languages and language in gold_languages:
                loose_per_language_f1[language].update(torch.ones(1),
                                                       torch.zeros(1))
                loose_per_language_mcc[language].update(torch.ones(1),
                                                        torch.zeros(1))
            elif language in predicted_languages:
                loose_per_language_f1[language].update(torch.zeros(1),
                                                       torch.ones(1))
                loose_per_language_mcc[language].update(torch.zeros(1),
                                                        torch.ones(1))
            elif language in gold_languages:
                loose_per_language_f1[language].update(torch.zeros(1),
                                                       torch.ones(1))
                loose_per_language_mcc[language].update(torch.zeros(1),
                                                        torch.ones(1))
            else:
                loose_per_language_f1[language].update(torch.zeros(1),
                                                       torch.zeros(1))
                loose_per_language_mcc[language].update(torch.zeros(1),
                                                        torch.zeros(1))
    return loose_accuracy, loose_per_language_f1, loose_per_language_mcc


def calculate_metrics(confusion_matrix, allowed_languages, predicted_languages_unique):
    per_langauge = {}

    for lang in allowed_languages:
        tn = 0
        gold_row_index = allowed_languages.index(lang)
        if lang in predicted_languages_unique:

            gold_column_index = predicted_languages_unique.index(lang)

            for row_index, row in enumerate(confusion_matrix):
                tn += sum([lang_value for column_index, lang_value in enumerate(row) if (row_index != gold_row_index) and (column_index != gold_column_index)])

            tp = confusion_matrix[gold_row_index][gold_column_index]
            fn = sum(confusion_matrix[gold_row_index][column_index] for column_index in range(len(confusion_matrix[gold_row_index])) if column_index != gold_column_index)
            fp = sum(confusion_matrix[row_index][gold_column_index] for row_index in range(len(confusion_matrix)) if row_index != gold_row_index)
        else:  # some language is never predicted, although should
            for row_index, row in enumerate(confusion_matrix):
                tn += sum([lang_value for lang_value in row if (row_index != gold_row_index)])
                tp = 0
                fn = sum(confusion_matrix[gold_row_index])
                fp = 0
        real_negatives = fp + tn
        fpr = fp / real_negatives if real_negatives > 0 else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_langauge[lang] = {
            "f1": f1, "tn": tn, "tp": tp, "fn": fn, "fp": fp, "fpr": fpr, "precision": precision, "recall": recall,
        }

    return per_langauge


# prints the confusion matrix, accuracy, precision, recall and macro f1-score
# calculates the loose scores -- if the correct language is in the list of predicted languages, the prediction is considered correct
def evaluate(args, identifier: AbstractLanguageIdentifier):
    logging.info(f"Evaluation of {args.method} started")
    logging.info(f"Loading dataset from {args.dataset}...")
    samples = [json.loads(line) for line in open(args.dataset)]

    supported_languages = identifier.languages

    # loose metrics
    loose_accuracy = torchmetrics.Accuracy("binary")
    loose_per_language_f1 = {language: torchmetrics.F1Score("binary") for
                             language in supported_languages}
    loose_per_language_mcc = {language: torchmetrics.MatthewsCorrCoef("binary")
                              for language in supported_languages}

    # strict metrics
    strict_accuracy = torchmetrics.Accuracy("binary")
    overlap_f1 = 0.0
    strict_per_language_f1 = {language: torchmetrics.F1Score("binary", ) for
                              language in supported_languages}
    strict_per_language_mcc = {
        language: torchmetrics.MatthewsCorrCoef("binary") for language in
        supported_languages}

    logging.info("Running inference...")

    start_time = time.time()
    logging.info(args)
    predicted_languages_unique = set()
    for i, sample in enumerate(tqdm(samples)):
        text = sample["text"]
        sample[GOLD_LANGUAGES] = set(sample["languages"])
        sample[PREDICTED_LANGUAGES] = set(identifier.identify(text))
        predicted_languages_unique.update(sample[PREDICTED_LANGUAGES])
    if predicted_languages_unique == set(supported_languages):
        predicted_languages_unique = supported_languages
    else:
        predicted_languages_unique = list(predicted_languages_unique)
        predicted_languages_unique.sort()
    end_time = time.time()

    logging.info("Calculating metrics...")
    confusion_matrix = [[0 for _ in predicted_languages_unique] for lang in
                        supported_languages]
    with open(f'{OUT_DIR}/{args.run_name}-predictions.jsonl',
              'w') as pred_file:
        for sample in samples:
            gold_languages = sample[GOLD_LANGUAGES]
            predicted_languages = sample[PREDICTED_LANGUAGES]
            for lang_gold in gold_languages:
                for lang in predicted_languages:
                    confusion_matrix[supported_languages.index(lang_gold)][
                        predicted_languages_unique.index(lang)
                    ] += 1

            loose_accuracy, loose_per_language_f1, loose_per_language_mcc = count_loose(
                predicted_languages,
                gold_languages,
                loose_accuracy,
                supported_languages,
                loose_per_language_f1,
                loose_per_language_mcc,
            )

            # strict metrics
            if predicted_languages == gold_languages:
                strict_accuracy.update(torch.ones(1), torch.ones(1))
            else:
                strict_accuracy.update(torch.zeros(1), torch.ones(1))

            common_languages = len(
                predicted_languages.intersection(gold_languages))
            overlap_precision = common_languages / len(predicted_languages)
            overlap_recall = common_languages / len(gold_languages)
            if overlap_precision + overlap_recall > 0:
                overlap_f1 += 2 * overlap_precision * overlap_recall / (
                        overlap_precision + overlap_recall)
            sample[GOLD_LANGUAGES] = list(sample[GOLD_LANGUAGES])
            sample[PREDICTED_LANGUAGES] = list(sample[PREDICTED_LANGUAGES])
            pred_file.write(json.dumps(sample) + '\n')
            for language in supported_languages:
                if language in predicted_languages and language in gold_languages:
                    strict_per_language_f1[language].update(torch.ones(1),
                                                            torch.ones(1))
                    strict_per_language_mcc[language].update(torch.ones(1),
                                                             torch.ones(1))
                else:

                    if language in predicted_languages:  # FP
                        strict_per_language_f1[language].update(torch.ones(1),
                                                                torch.zeros(1))
                        strict_per_language_mcc[language].update(torch.ones(1),
                                                                 torch.zeros(
                                                                     1))
                    elif language in gold_languages:  # FN

                        strict_per_language_f1[language].update(torch.zeros(1),
                                                                torch.ones(1))
                        strict_per_language_mcc[language].update(
                            torch.zeros(1), torch.ones(1))
                    else:
                        strict_per_language_f1[language].update(torch.zeros(1),
                                                                torch.zeros(1))
                        strict_per_language_mcc[language].update(
                            torch.zeros(1), torch.zeros(1))
    per_language = calculate_metrics(confusion_matrix, supported_languages, predicted_languages_unique)
    draw_confusion_matrix(confusion_matrix, supported_languages, predicted_languages_unique, args)
    logging.info(f"\n# Results for {args.method}:\n")
    logging.info("## Loose metrics")
    logging.info(f"\tLoose accuracy: {loose_accuracy.compute().item():.2%}")
    logging.info(
        f"\tLoose macro F1: {sum([loose_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    logging.info(
        f"\tLoose macro MCC: {sum([loose_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")

    logging.info("### Per-language metrics")
    for language in supported_languages:
        logging.info(f"\t{language}:")
        logging.info(
            f"\t\tF1: {loose_per_language_f1[language].compute().item():.2%}")
        logging.info(
            f"\t\tMCC: {loose_per_language_mcc[language].compute().item():.2%}")
        logging.info(per_language[language])

    logging.info("\n\n## Strict metrics")
    logging.info(f"\tStrict accuracy: {strict_accuracy.compute().item():.2%}")
    logging.info(f"\tOverlap F1: {overlap_f1 / len(samples):.2%}")
    logging.info(
        f"\tStrict macro F1: {sum([strict_per_language_f1[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    logging.info(
        f"\tStrict macro MCC: {sum([strict_per_language_mcc[language].compute().item() for language in supported_languages]) / len(supported_languages):.2%}")
    logging.info("### Per-language metrics")
    for language in supported_languages:
        logging.info(f"\t{language}:")
        logging.info(
            f"\t\tF1: {strict_per_language_f1[language].compute().item():.2%}")
        logging.info(
            f"\t\tMCC: {strict_per_language_mcc[language].compute().item():.2%}")

    logging.info("\n\n## CPU inference time")
    logging.info(f"\tTotal runtime: {end_time - start_time:.2f} seconds")
    logging.info(
        f"\tms / sentence: {(end_time - start_time) / len(samples) * 1000:.2f} ms")


def main():
    random.seed(42)
    torch.manual_seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    args = parse_args()
    os.makedirs(os.path.join(OUT_DIR, args.model, args.dataset), exist_ok=True)
    if not args.run_name:
        args.run_name = f"{args.method}_{args.model}-k-{args.k}-{args.threshold}-on-{args.dataset}".replace(
            '/', ''
        )
    is_bert = args.method == "bert"
    if is_bert:
        args.run_name += f"-other_if_below_threshold-{args.other_if_below_threshold}"
    args.log_fn = os.path.join(OUT_DIR, args.run_name)
    print(args.log_fn)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"{args.log_fn}.log",
        force=True,
    )

    if args.method == "random":
        identifier = RandomLanguageIdentifier(args)
    elif args.method == "fasttext_hf_hub":
        identifier = FasttextHfHubIdentifier(args, languages=args.languages.split(','), repo_id=args.model)
    elif args.method == "gpt2":
        identifier = GPT2Identifier(args)
    elif args.method == "fasttext":
        identifier = FasttextLanguageIdentifier(args)
    elif is_bert:
        identifier = BERTIdentifier(args)
    elif args.method == 'mlp':
        identifier = MlpIdentifier(args)
    elif args.method == 'fasttext_ensemble':
        identifier = FasttextEnsembleIdentifier(args, languages=args.languages.split(','))
    evaluate(args, identifier)


if __name__ == "__main__":
    main()
