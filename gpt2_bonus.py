import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from model import GPT
from utils import set_seed
from bpe import BPETokenizer
set_seed(3407)

use_mingpt = True # use minGPT or huggingface/transformers model?
model_type = 'gpt2'
device = 'cuda'

if use_mingpt:
    model = GPT.from_pretrained(model_type)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id # suppress a warning

# ship model to device and set to eval mode
model.to(device)
model.eval()

def generate(prompt='', num_samples=10, steps=20, do_sample=True):

    # tokenize the input prompt into integer input sequence
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            # to create unconditional samples...
            # manually create a tensor with only the special <|endoftext|> token
            # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
            x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
        else:
            x = tokenizer(prompt).to(device)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        if prompt == '':
            # to create unconditional samples...
            # huggingface/transformers tokenizer special cases these strings
            prompt = '<|endoftext|>'
        encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
        x = encoded_input['input_ids']

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*76)
        print(out)

generate(prompt='SpongeBob SquarePants', num_samples=5, steps=20)
generate(prompt='The big', num_samples=5, steps=20)
generate(prompt='Please forgive', num_samples=5, steps=20)
generate(prompt='Apple pie', num_samples=5, steps=20)
generate(prompt='Pizza delivery', num_samples=5, steps=20)
generate(prompt='The best', num_samples=5, steps=20)
generate(prompt='Prime Minister', num_samples=5, steps=20)
generate(prompt='The queen', num_samples=5, steps=20)
generate(prompt='Best singer', num_samples=5, steps=20)
generate(prompt='My name', num_samples=5, steps=20)
