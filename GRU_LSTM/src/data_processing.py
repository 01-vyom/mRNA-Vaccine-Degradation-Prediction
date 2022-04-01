import numpy as np

pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    base_fea = np.transpose(
        np.array(
            df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
        ),
        (0, 2, 1),
    )
    bpps_sum_fea = np.array(df["bpps_sum"].to_list())[:, :, np.newaxis]
    bpps_max_fea = np.array(df["bpps_max"].to_list())[:, :, np.newaxis]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list())[:, :, np.newaxis]
    bpps_v_fea = np.array(df["bpps_v"].to_list())[:, :, np.newaxis]
    bpps_m_fea = np.array(df["bpps_m"].to_list())[:, :, np.newaxis]
    return np.concatenate(
        [base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea, bpps_v_fea, bpps_m_fea], 2
    )
    return base_fea


# additional features


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"./bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"./bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr


def read_bpps_nb(df):
    # normalized non-zero number
    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn
    bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914  # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"./bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr


def read_bpps_m(df):
    e = 0.00000001
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"./bpps/{mol_id}.npy")
        vec = []
        for i in range(bpps.shape[0]):
            m = 0
            l = 0
            for j in range(bpps.shape[0]):
                if bpps[i][j] > 0:
                    m = m + (j * bpps[i][j])
                    l = l + 1
            m = m / (l + e)
            vec.append(m)
        bpps_arr.append(vec)
    return bpps_arr


def read_bpps_v(df):
    b = 0.9  # beta for exponential weaghted average with bias correction
    e = 0.00000001
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"./bpps/{mol_id}.npy")
        vec = []
        for i in range(bpps.shape[0]):
            v = 0
            m = 0
            l = 0
            for j in range(bpps.shape[0]):
                if bpps[i][j] > 0:
                    v = b * v + (1 - b) * bpps[i][j]
                    m = m + v
                    l = l + 1
            m = m / (l + e)
            vec.append(m)
        bpps_arr.append(vec)
    return bpps_arr


def augmentation(df, aug_df):
    # from https://www.kaggle.com/code/its7171/how-to-generate-augmentation-data
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
