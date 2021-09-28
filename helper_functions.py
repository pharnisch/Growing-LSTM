def get_data(amount=-1):
    f = open("./data/conll-2003-english/train.txt", "r")
    data = []
    sentence_tokens = []
    sentence_labels = []
    sent_has_to_be_saved = False
    for line in f.read().split("\n")[2:]:
        line_parts = line.split(" ")
        if len(line_parts) == 1:
            if sent_has_to_be_saved:
                data.append((sentence_tokens, sentence_labels))
                sentence_tokens = []
                sentence_labels = []
                sent_has_to_be_saved = False
                if len(data) == amount:
                    return data
        else:
            sentence_tokens.append(line_parts[0])
            sentence_labels.append(line_parts[3])
            sent_has_to_be_saved = True
    if sent_has_to_be_saved:
        data.append((sentence_tokens, sentence_labels))
    return data


def get_dicts(data):
    word_to_ix = {"O": 0}
    label_to_ix = {"O": 0}
    threshold = 2
    word_counts = {}
    # 1. count all words
    for (sentence, sentence_labels) in data:
        for word in sentence:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        for label in sentence_labels:
            if label not in label_to_ix:
                label_to_ix[label] = len(label_to_ix)
    # 2. insert all into dict with cound >= threshold
    for word in word_counts:
        if word_counts[word] >= threshold:
            word_to_ix[word] = len(word_to_ix)
    return word_to_ix, label_to_ix


def prepare_sequence(seq, to_ix):
    import torch
    idxs = []
    for element in seq:
        if element in to_ix:
            idxs.append(to_ix[element])
        else:
            idxs.append(0)
    return torch.tensor(idxs, dtype=torch.long)


# data = get_data()
# word_to_ix, label_to_ix = get_dicts(data)
# seq = prepare_sequence(data[0][0], word_to_ix)
# seq_labels = prepare_sequence(data[0][1], label_to_ix)
# print(data[0])
# print(seq)
# print(seq_labels)