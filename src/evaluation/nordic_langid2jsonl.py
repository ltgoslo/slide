#! /usr/bin/env python3

import json

label_rewrite = {"fo": "other", "is": "other", "dk": "da"}

def convert(infilename, outfilename):
	with open(infilename, 'r', encoding="utf-8") as infile:
		with open(outfilename, 'w', encoding="utf-8") as outfile:
			for line in infile:
				tokens = line.strip().split(" ")
				label = tokens[-1]
				text = " ".join(tokens[1:-1])
				if label not in ("dk", "sv", "nb", "nn", "fo", "is"):
					print("Unknown label:", label)
					print(text)
					continue
				newlabel = label_rewrite.get(label, label)
				item = json.dumps({"text": text.strip(), "languages": [newlabel]}, ensure_ascii=False)
				outfile.write(item + "\n")


if __name__ == "__main__":
	convert("nordic_dsl_10000test.csv", "nordic_dsl_test10k.jsonl")
	convert("nordic_dsl_50000test.csv", "nordic_dsl_test50k.jsonl")
