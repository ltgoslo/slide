from collections import defaultdict
import pandas as pd
import os
import json

mapping = {'hr': 'hrv_Latn', "bs": "bos_Latn", "sr": "srp_Latn"}
# get the data from https://zenodo.org/records/10998042
# those on the shared task github are w/o gold labels for the test
data = pd.read_csv(os.path.expanduser(r"~/Downloads/VarDial2024_DSL-ML_BCMS/test.tsv"), sep='\t', header=None)
counter = defaultdict(int)
with open("hbs_test.jsonl", "w") as f:
    for line in data.iterrows():
        line = line[1]
        languages = []
        for lang in line[0].split(','):
            if lang != "me":
                languages.append(mapping[lang])
                counter[mapping[lang]] += 1
            else:
                if "sr" not in line[0]:
                    languages.append("srp_Latn")
                    counter["srp_Latn"] += 1
        out_line = {"text": line[1], "languages": languages}
        if out_line["languages"]:
            f.write(json.dumps(out_line) + '\n')
print(counter)
