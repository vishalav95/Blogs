You will understand how to use the ‘generate’ method to accomplish common tasks such as [language generation](https://huggingface.co/docs/transformers/tasks/language_modeling), [prompting](https://huggingface.co/Meli/GPT2-Prompt), [translation](https://huggingface.co/docs/transformers/tasks/translation), and [summarization](https://huggingface.co/docs/transformers/tasks/summarization), using greedy search, beam search decoding methods and sampling methods. The ‘generate’ method’s parameters can be tweaked to accomplish the above-mentioned tasks.

The ‘generate’ method is a part of the ‘Transformer’ class which generates a sequence of token ids for models with a language modelling head. It can be used to generate text for text-decoder, text-to-text-, speech-to-text, and vision-to-text models. You can use this method to achieve a variety of results ranging from generating text to summarising text, translating text from one language to another and so on. 

In the consequent examples, you will see which parameters need to be tweaked to achieve specific results.

Prerequisites
Before running the examples, install the ‘transformers’ and ‘tensorflow’ library.
pip install transformers

```
pip install tensorflow==2.1
```

Also add the EOS token as PAD token to avoid warnings in your environment as shown below:

```
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
```

## Greedy decoding/Greedy search
Note: Describe greedy decoding, along with the drawbacks. Also add images to describe how the probability is taken into account to choose the next word.
Greedy search is used in language generation.

You can implement greedy decoding using greedy_search() by setting the below parameters:
```
num_beams=1
do_sample=False
```
```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Today I can finally"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# generate up to 100 tokens
outputs = model.generate(input_ids, num_beams=1, do_sample=False, max_length=100)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

**Output:**

['Today I can finally say that I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the results of my research. I am very happy with the']


In the above output, point out at the redundant text generated, which would serve as a reason to jump to beam search decoding.
Beam search decoding
Note: Describe beam search decoding, how it has a leverage over greedy_search(), along with the drawbacks. Also add images to describe how the probability product is taken into account to choose the next word.

Beam search is used in machine translation, and summarization.

You can implement beam search decoding using beam_search() by setting the below parameters:
```
num_beams>1
do_sample=False
```
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

sentence = "Paris is one of the densest populated areas in Europe."
input_ids = tokenizer(sentence, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

Tip: If you encounter “ValueError: This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed in order to use this tokenizer.”, ensure you follow the below steps:
```
pip install sentencepiece
```

After installing this package, restart your kernel/runtime. 
[Reference](https://stackoverflow.com/questions/65431837/transformers-v4-x-convert-slow-tokenizer-to-fast-tokenizer)

**Output:**

['Paris ist eines der dichtesten besiedelten Gebiete Europas.']



### Multinomial sampling
Describe multinomial sampling and its applications, along with the drawbacks.
You can implement multinomial sampling using sample() by setting the below parameters:
```
num_beams=1
do_sample=True
```
```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Today I had"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# sample up to 45 tokens
torch.manual_seed(0)
outputs = model.generate(input_ids, do_sample=True, max_length=45)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

**Output:**

['Today I had a wonderful time with our family...']



The above code examples were run on Google Colab.
