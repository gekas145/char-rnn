from unidecode import unidecode 
import pandas as pd
import zipfile

lines = []

with zipfile.ZipFile("data/sherlock.zip", "r") as zf:
    for file in zf.namelist():
        with zf.open(file, "r") as f:
            lines += f.readlines()

text = b" ".join(lines).decode()
text = unidecode(text)
vocab = list(set(text))
vocab.sort()
vocab = "".join(vocab)

print(vocab)
print(len(vocab))

n = 0
length = 50

context = []
label = []

while n + length < len(text):
    context.append(text[n : n+length])
    label.append(text[n+1 : n+length+1])

    n += length + 1


pd.DataFrame({"context": context, "label": label}).sample(frac=1).to_parquet("data/next-character.parquet", index=False)









