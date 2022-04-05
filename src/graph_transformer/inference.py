import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from src.graph_transformer.data_processing import *
from src.graph_transformer.model_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
columns = [
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]


def predict(
    X_node,
    X_node_pub,
    X_node_pri,
    adjacency_matrix_pub,
    adjacency_matrix_pri,
    seq_len_target,
    targets,
    test_pub,
    test_pri,
    n_folds=5,
    model_path="models.h5",
    sub_file="submission.csv",
):
    p_pub = 0
    p_pri = 0
    base = get_base(X_node, adjacency_matrix_pub)
    model = get_model(base, X_node, adjacency_matrix_pub, seq_len_target)
    for fold in range(n_folds):
        filepath_list = model_path.split(".")
        filepath = f"./{filepath_list[0]}_{fold}.{filepath_list[1]}"
        model.load_weights(filepath)
        p_pub += model.predict([X_node_pub, adjacency_matrix_pub]) / n_folds
        p_pri += model.predict([X_node_pri, adjacency_matrix_pri]) / n_folds

    for i, target in enumerate(targets):
        test_pub[target] = [list(p_pub[k, :, i]) for k in range(p_pub.shape[0])]
        test_pri[target] = [list(p_pri[k, :, i]) for k in range(p_pri.shape[0])]

    preds_ls = []
    for df, preds in [(test_pub, p_pub), (test_pri, p_pri)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]

            single_df = pd.DataFrame(single_pred, columns=targets)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

            preds_ls.append(single_df)

    preds_df = pd.concat(preds_ls)
    preds_df.to_csv(sub_file, index=False)
    preds_df.head()
    print(f"wrote to submission file")
    return preds_df


def model_average(pred_with_ae, pred_without_ae, sub_file="submission_average.csv"):
    preds_df_new = pred_without_ae[columns] * 0.5 + pred_with_ae[columns] * 0.5
    preds_df_new["id_seqpos"] = pred_without_ae["id_seqpos"]
    preds_df_new.to_csv(sub_file, index=False)
    preds_df_new.head()
    print(f"wrote to submission file")


def __main__():
    denoise = True  # if True, use train data whose signal_to_noise > 1
    path = "."
    allocate_gpu_memory()
    train = pd.read_json(f"{path}/train.json", lines=True)
    if denoise:
        train = train[train.signal_to_noise > 1].reset_index(drop=True)

    test = pd.read_json(f"{path}/test.json", lines=True)
    test_pub = test[test["seq_length"] == 107]
    test_pri = test[test["seq_length"] == 130]
    sub = pd.read_csv(f"{path}/sample_submission.csv")

    targets = list(sub.columns[1:])
    print(targets)

    y_train = []
    seq_len = train["seq_length"].iloc[0]
    seq_len_target = train["seq_scored"].iloc[0]
    ignore = -10000
    ignore_length = seq_len - seq_len_target
    for target in targets:
        y = np.vstack(train[target])
        dummy = np.zeros([y.shape[0], ignore_length]) + ignore
        y = np.hstack([y, dummy])
        y_train.append(y)
    y = np.stack(y_train, axis=2)
    y.shape

    adjacency_matrix_pub = get_adj_matrix(test_pub)
    adjacency_matrix_pri = get_adj_matrix(test_pri)

    X_node = get_node_features(train)
    X_node_pub = get_node_features(test_pub)
    X_node_pri = get_node_features(test_pri)

    n_folds = 7
    model_path = "model_with_ae.h5"
    sub_file = "result/submission_with_ae.csv"
    pred_with_ae = predict(
        X_node,
        X_node_pub,
        X_node_pri,
        adjacency_matrix_pub,
        adjacency_matrix_pri,
        seq_len_target,
        targets,
        test_pub,
        test_pri,
        n_folds=n_folds,
        model_path=model_path,
        sub_file=sub_file,
    )

    model_path = "model_without_ae.h5"
    sub_file = "result/submission_without_ae.csv"
    pred_without_ae = predict(
        X_node,
        X_node_pub,
        X_node_pri,
        adjacency_matrix_pub,
        adjacency_matrix_pri,
        seq_len_target,
        targets,
        test_pub,
        test_pri,
        n_folds=n_folds,
        model_path=model_path,
        sub_file=sub_file,
    )
    sub_file = "result/submission_without_with_ae_average.csv"
    model_average(pred_with_ae, pred_without_ae, sub_file)

