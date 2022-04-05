import numpy as np
from tqdm.notebook import tqdm

def get_adj_matrix(data_df, path):
    As = []
    for id in tqdm(data_df["id"]):
        a = np.load(f"{path}/bpps/{id}.npy")
        As.append(a)
    As = np.array(As)

    ## get adjacent matrix from structure sequence
    sequence_structure_adj = []
    for i in tqdm(range(len(data_df))):
        seq_length = data_df["seq_length"].iloc[i]
        structure = data_df["structure"].iloc[i]
        sequence = data_df["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U"): np.zeros([seq_length, seq_length]),
            ("C", "G"): np.zeros([seq_length, seq_length]),
            ("U", "G"): np.zeros([seq_length, seq_length]),
            ("U", "A"): np.zeros([seq_length, seq_length]),
            ("G", "C"): np.zeros([seq_length, seq_length]),
            ("G", "U"): np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1

        a_strc = np.stack([a for a in a_structures.values()], axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)
        sequence_structure_adj.append(a_strc)

    sequence_structure_adj = np.array(sequence_structure_adj)
    print(sequence_structure_adj.shape)

    ## adjacent matrix based on distance on the sequence
    ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4
    idx = np.arange(As.shape[1])
    distance_matrix = []
    for i in range(len(idx)):
        distance_matrix.append(np.abs(idx[i] - idx))

    distance_matrix = np.array(distance_matrix) + 1
    distance_matrix = 1 / distance_matrix
    distance_matrix = distance_matrix[None, :, :]
    distance_matrix = np.repeat(distance_matrix, len(As), axis=0)

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(distance_matrix ** i)
    distance_matrix = np.stack(Dss, axis=3)
    print(distance_matrix.shape)

    adjacency_matrix = np.concatenate(
        [As[:, :, :, None], sequence_structure_adj, distance_matrix], axis=3
    ).astype(np.float32)

    return adjacency_matrix


def get_node_features(train):
    ## get node features, which is one hot encoded
    mapping = {}
    vocab = ["A", "G", "C", "U"]
    for i, s in enumerate(vocab):
        mapping[s] = [0] * len(vocab)
        mapping[s][i] = 1
    X_node = np.stack(
        train["sequence"].apply(lambda x: list(map(lambda y: mapping[y], list(x))))
    )

    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = [0] * len(vocab)
        mapping[s][i] = 1
    X_loop = np.stack(
        train["predicted_loop_type"].apply(
            lambda x: list(map(lambda y: mapping[y], list(x)))
        )
    )

    X_node = np.concatenate([X_node, X_loop], axis=2)

    ## interaction
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis=2)
    vocab = sorted(set(a.flatten()))
    print(vocab)
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis=2)
    X_node = np.concatenate([X_node, ohes], axis=2).astype(np.float32)

    print(X_node.shape)
    return X_node


def augmentation(df, aug_df):
    target_df = df.copy()
    new_df = aug_df[aug_df["id"].isin(target_df["id"])]

    del target_df["structure"]
    del target_df["predicted_loop_type"]
    new_df = new_df.merge(target_df, on=["id", "sequence"], how="left")

    df["cnt"] = df["id"].map(new_df[["id", "cnt"]].set_index("id").to_dict()["cnt"])
    df["log_gamma"] = 100
    df["score"] = 1.0
    df = df.append(new_df[df.columns])
    return df
