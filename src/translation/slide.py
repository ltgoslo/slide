import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gzip


device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='/cluster/work/users/mariiaf/slide/')
parser.add_argument('--data', default='/cluster/work/users/mariiaf/glotlid-corpus/v3.1/dan_Latn/dan_Latn_GlotStoryBook.txt.gz')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()
print(args)
dataset = load_dataset("text", data_files=args.data)['train']
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
model.eval()
prompt_nn = """{source_text}
Nynorsk:"""

prompt_nb = """{source_text}
Bokmål:"""

prompt_da = """{source_text}
Dansk:"""

prompt_sv = """{source_text}
Svenska:"""


eos_token_ids = [
    i for i in range(tokenizer.vocab_size)
    if 'Ċ' in tokenizer.convert_ids_to_tokens(i)
]
print(eos_token_ids)

print([tokenizer.convert_ids_to_tokens(i) for i in eos_token_ids])
LANG_PROMPTS = {
    'dan_Latn': (
        prompt_nn,
        prompt_nb,
        prompt_sv,
    ),
    'nno_Latn': (
        prompt_da,
        prompt_nb,
        prompt_sv,
    ),
    'nob_Latn': (
        prompt_nn,
        prompt_da,
        prompt_sv,
    ),
    'swe_Latn': (
        prompt_nn,
        prompt_da,
        prompt_nb,
    )
}

lang = args.data.split('/')[-2]
tokenizer.pad_token = tokenizer.eos_token
for prompt in LANG_PROMPTS[lang]:
    lang_name = prompt.split("\n")[-1][:-1]
    print(f"Predicting {lang_name}")
    outfile = args.data + f'-translations-{lang_name}.txt.gz'
    
    column_name = f'text_{lang_name}'
    dataset = dataset.map(lambda x: {column_name: [prompt.format(**{'source_text':text}) for text in x['text']]}, batched=True)
    test_iter = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
    )
    
    with gzip.open(outfile, 'wt', encoding='utf8') as out:
        for inp  in tqdm(test_iter):
            inp = tokenizer(inp[column_name], truncation=False, padding='longest', return_tensors='pt')
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inp['input_ids'].to(device),
                    attention_mask=inp['attention_mask'].to(device),
                    max_new_tokens=inp['input_ids'].size(1),
                    do_sample=False,
                    eos_token_id=eos_token_ids,
                )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for output in outputs:
                out.write(output + '\n')
