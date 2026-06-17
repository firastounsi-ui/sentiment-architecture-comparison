import re
from collections import Counter


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    """
    Simple tokenizer:
    - lowercase text
    - keep words and punctuation as tokens
    """
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:
    def __init__(self, max_size=20000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq

        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

        self.id_to_token = {
            0: PAD_TOKEN,
            1: UNK_TOKEN,
        }

    def build(self, texts):
        counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        most_common = counter.most_common(self.max_size)

        for token, freq in most_common:
            if freq < self.min_freq:
                continue

            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, text, max_length):
        tokens = tokenize(text)

        ids = [
            self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])
            for token in tokens
        ]

        ids = ids[:max_length]

        while len(ids) < max_length:
            ids.append(self.token_to_id[PAD_TOKEN])

        return ids

    def __len__(self):
        return len(self.token_to_id)


if __name__ == "__main__":
    texts = [
        "This movie was great!",
        "This movie was terrible.",
        "I loved this movie."
    ]

    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.build(texts)

    print("Vocab size:", len(vocab))
    print("Token to ID:", vocab.token_to_id)

    encoded = vocab.encode("This movie was unknownword!", max_length=8)
    print("Encoded:", encoded)
