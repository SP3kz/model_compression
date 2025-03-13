import re

def tokenize_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return list(set(words))  # Return unique words

text = "This is a test sentence for tokenization."
tokens = tokenize_text(text)
print("Tokenized Output:", tokens)
