def get_sequence_edge_pair(sequences, lengths, padding_idx, window_size):
    edge_pair_set = set()
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        if (i + 1) % 1000 == 0:
            print("get_edge_pair: {}/{} {}%".format(i + 1, len(sequences), (i + 1) * 100 / len(sequences)))
        seq = [padding_idx] * window_size + seq + [padding_idx] * window_size
        for j in range(window_size, len(seq) - window_size):
            master = seq[j]
            for k in range(j - window_size, j + window_size + 1):
                if j == k:
                    continue
                slave = seq[k]
                master, slave = min(master, slave), max(master, slave)
                if slave == padding_idx:
                    continue
                pair_key = "{}+{}".format(master, slave)
                edge_pair_set.add(pair_key)
    return edge_pair_set


def get_Semantic_edge_pair():
    pass


if __name__ == '__main__':
    import numpy as np

    num_words = 10000
    padding_idx = num_words
    sequences = np.random.randint(0, num_words, (10000, 300)).tolist()
    a = get_sequence_edge_pair(sequences, padding_idx, 2)
    print(len(a))
