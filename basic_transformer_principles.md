# Tokenizer

Pretend there is a 50000 vocab size and 3 dimension embedding.

```python
sentence = 'Life is short, eat dessert first'
​
dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}

print(dc)
```

```python

import torch
​
sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)
print(sentence_int)
```

Now, use torch nn embedding to make an embedding tensor
```python
vocab_size = 50_000
​
torch.manual_seed(123)
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()
​
print(embedded_sentence)
print(embedded_sentence.shape)
```

