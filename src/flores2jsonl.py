import argparse
from collections import defaultdict
import json

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', default="openlid-v3-evaluation/src/evaluation/data/flores_plus/devtest.jsonl")
args = parser.parse_args()
counter = defaultdict(int)
with open("flores_plus_devtest.jsonl", 'w') as out:
    with open(args.path, 'r') as f:
        for line in tqdm(f):
            line = json.loads(line)
            out_line = {"text": line["text"], "languages": []}
            if line["iso_639_3"] == "nob":
                out_line["languages"].append("nb")
                counter["nb"] += 1
            if line["iso_639_3"] == "nno":
                out_line["languages"].append("nn")
                counter["nn"] += 1
            if line["iso_639_3"] == "dan":
                out_line["languages"].append("da")
                counter["da"] += 1
            if line["iso_639_3"] == "swe":
                out_line["languages"].append("sv")
                counter["sv"] += 1
            if not out_line["languages"]:
                out_line["languages"].append("other")
            out.write(json.dumps(out_line) + '\n')
print(counter)
