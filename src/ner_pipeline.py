"""
NOTE: I don't think this will run on Fox because installing spacy models with pip.

This script performs named entity recognition (NER) on text data using spaCy 
language models. It takes a file containing text data as input and outputs CSV 
files containing the extracted named entities for each specified language.

Usage:
	ner_pipeline.py [-h] [--output_dir OUTPUT_DIR] [--languages langs [langs ...]] [--model_size {sm,md,lg}] data
Arguments:
	data                Filepath to the data to be tagged.
Options:
  	-h, --help          show this help message and exit
  	--output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save output dataframes.
  	--languages langs [langs ...], -l langs [langs ...]
                        List of languages to be used for tagging.
  	--model_size {sm,md,lg}, -s {sm,md,lg}
                        Size of the spacy model to use for tagging.
        
The script installs and loads the specified spaCy language models. It then 
extracts named entities from the text data for each language using the 
corresponding model. The extracted entities are saved as CSV files in the 
specified output directory. Logging is saved to an out file.

NOTE: The script requires the spaCy library and the smart_open library for file handling.
"""

import pandas as pd
import spacy
import json
import logging
import subprocess

from argparse import ArgumentParser
from pathlib import Path
from smart_open import open
from tqdm import tqdm
from uuid import uuid4

# Templates for spacy models given language
# NOTE: Nynorsk does not have its own model included with spacy
nlp_model_names = {
    "nn": "nb_core_news_{size}",
    "nb": "nb_core_news_{size}",
    "da": "da_core_news_{size}",
    "sv": "sv_core_news_{size}",
    # "other": "xx_ent_wiki_{size}",
    # "other": "xx_sent_ud_sm"
}


def model_install_and_load(model: str) -> spacy.Language:
    """
    Install and load a spaCy language model.
    Args:
            model (str): The name of the spaCy language model to install and load.
    Returns:
            spacy.Language: The loaded spaCy language model.
    """
    if not spacy.util.is_package(model):
        logger.warning(f"Model '{model}' not found. Installing...")
        subprocess.run(["python3", "-m", "spacy", "download", model])

    return spacy.load(model)


def ner_by_language(
    df: pd.DataFrame, language: str, nlp: spacy.Language
) -> pd.DataFrame:
    """
    Extracts named entities from the given DataFrame using the specified model.
    Args:
            df (pd.DataFrame): The subsest of the DataFrame containing the data for
                                                    the given language.
            language (str): The language of the text data.
            nlp (spacy.Language): The Spacy language model.
    Returns:
            pd.DataFrame: A DataFrame containing the extracted named entities along
                            with their labels and language.
                    - columns=["id", "text", "entity", "label", "language", "start_char", "end_char"]
    """
    res = pd.DataFrame(
        columns=["id", "text", "entity", "label", "language", "start_char", "end_char"]
    )
    progressbar = tqdm(total=len(df))
    for i, row in df.iterrows():
        doc = nlp(row["text"])
        if doc.ents:
            for ent in doc.ents:
                res.loc[len(res)] = [
                    i,
                    row["text"],
                    ent.text,
                    ent.label_,
                    language,
                    ent.start_char,
                    ent.end_char,
                ]
        progressbar.update(1)
    return res


def standardize_labels(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Maps labels to one of the following 4: ['LOC', 'ORG', 'PERS', 'MISC']

    Args:
            df (pd.DataFrame): the language specific instances
            language (str): the language of the data,
                            one of ["da", "nb", "nn", "sv", "other"]

    Returns:
            pd.DataFrame: the data with standardized tags.
    """
    if language in ["nb", "nn"]:
        df["label"] = df["label"].replace("GPE_LOC", "LOC")
        df["label"] = df["label"].replace("GPE_ORG", "ORG")
        df["label"] = df["label"].replace("PROD", "MISC")
        df["label"] = df["label"].replace("DRV", "MISC")
        df["label"] = df["label"].replace("EVT", "MISC")
    elif language == "sv":
        df["label"] = df["label"].replace("PRS", "PER")
        df["label"] = df["label"].replace("MSR", "MISC")
        df["label"] = df["label"].replace("TME", "MISC")
        df["label"] = df["label"].replace("WRK", "MISC")
        df["label"] = df["label"].replace("OBJ", "MISC")
        df["label"] = df["label"].replace("EVN", "MISC")

    return df


def main(args):
    logger.info("Loading in the models...")
    nlp_models = {
        lang: model_install_and_load(model.format(size=args.model_size))
        for lang, model in nlp_model_names.items()
        if lang in args.languages
    }
    nlp_models["other"] = model_install_and_load("xx_sent_ud_sm")

    df = pd.DataFrame(
        [json.loads(line) for line in open(args.data, "r", encoding="utf-8")]
    )
    entities = {}
    for language in args.languages:
        lang_subset = df[df["languages"].apply(lambda x: language in x)]
        nlp = nlp_models.get(language)
        logger.info(f"Tagging '{language}'...")
        entities[language] = ner_by_language(lang_subset, language, nlp)

    if args.output_dir is not None:
        for language, df in entities.items():
            df = standardize_labels(df, language)
            df.to_json(
                args.output_dir / Path(f"{language}_{args.name}_ne.csv"),
                orient="records",
                lines=True,
            )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data", type=str, help="Filepath to the data to be tagged.")
    parser.add_argument(
        "--output_dir", "-o", type=str, help="Directory to save output dataframes."
    )
    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        metavar="langs",
        default=["nn", "nb", "da", "sv", "other"],
        help="List of languages to be used for tagging.",
    )
    parser.add_argument(
        "--model_size",
        "-s",
        type=str,
        choices=["sm", "md", "lg"],
        help="Size of the spacy model to use for tagging.",
    )

    args = parser.parse_args()

    args.data = Path(args.data)
    assert args.data.exists(), "Path to data file does not exist"

    # Output directory checks and formatting
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
        if args.output_dir.exists():
            assert args.output_dir.is_dir(), "Output path must be a directory."
        else:
            args.output_dir.mkdir()
    else:
        args.output_dir = Path()

    # Saving input file stem for output files
    args.name = args.data.stem
    if "." in args.name:
        args.name = args.name.split(".")[0]

    # For displaying progress
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=args.output_dir / (str(uuid4().hex) + ".out"),
        filemode="a",
        format="%(levelname)s: %(asctime)s: %(message)s",
        level=logging.INFO,
    )

    # Checking model size
    if args.model_size is None:
        logger.info("No model size specified. Defaulting to 'sm'.")
        args.model_size = "sm"

    logger.info("ARGUMENTS:")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    main(args)
