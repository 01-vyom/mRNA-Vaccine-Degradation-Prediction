import numpy as np
import pandas as pd
from src.GRU_LSTM.data_processing import *
from src.GRU_LSTM.model_utility import *
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
allocate_gpu_memory()
Ver = "GRU_LSTM_10_"
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def train_model(train, type=0, FOLD_N=5):
    print("Model Type", type + 1, ":")
    gkf = GroupKFold(n_splits=FOLD_N)
    fold_val_losses = []
    holdouts = []
    holdout_preds = []
    histories = []
    plot_labels = []
    for cv, (tr_idx, vl_idx) in enumerate(
        gkf.split(train, train["reactivity"], train["cluster_id"])
    ):
        print("Fold ", cv + 1, "/", FOLD_N, ": Training Started...")
        trn = train.iloc[tr_idx]
        x_trn = preprocess_inputs(trn)
        y_trn = np.array(trn[pred_cols].values.tolist()).transpose((0, 2, 1))
        w_trn = np.log(trn.signal_to_noise + 1.1) / 2

        val = train.iloc[vl_idx]
        x_val_all = preprocess_inputs(val)
        val = val[val.SN_filter == 1]
        x_val = preprocess_inputs(val)
        y_val = np.array(val[pred_cols].values.tolist()).transpose((0, 2, 1))

        model = build_model(type=type)

        history = model.fit(
            x_trn,
            y_trn,
            validation_data=(x_val, y_val),
            batch_size=64,
            epochs=60,
            sample_weight=w_trn,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(),
                tf.keras.callbacks.ModelCheckpoint(f"./model{Ver}{type}_cv{cv}.h5"),
            ],
        )
        print(
            f"Min training loss={min(history.history['loss'])}, min validation loss={min(history.history['val_loss'])}"
        )
        histories.append(history)

        model.load_weights(f"/model{Ver}{type}_cv{cv}.h5")
        holdouts.append(train.iloc[vl_idx])
        holdout_preds.append(model.predict(x_val_all))
        fold_val_losses.append(min(history.history["val_loss"]))
    for cv in range(FOLD_N):
        plt.plot(histories[cv].history["loss"])
        plt.plot(histories[cv].history["val_loss"])
        plot_labels.append("train_loss_" + str(cv))
        plot_labels.append("val_loss_" + str(cv))
    plt.title("Training History")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("epoch")
    plt.legend(
        plot_labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="x-small",
    )
    plt.savefig(
        "./result/" + Ver + "_Model_" + str(type) + ".png", bbox_inches="tight", dpi=600
    )
    plt.show()
    return holdouts, holdout_preds, fold_val_losses


def run(train, nmodel=4, FOLD_N=5, debug=False):
    val_df, val_preds, model_losses = [], [], []
    if debug:
        nmodel = 1
    for i in range(nmodel):
        holdouts, holdout_preds, fold_losses = train_model(train, i, FOLD_N)
        val_df += holdouts
        val_preds += holdout_preds
        model_losses.append(fold_losses)
    return val_df, val_preds, model_losses


def __main__():
    aug_data = "./data/augmented_data.csv"
    train_data = "./train.json"
    debug = False
    train = pd.read_json(train_data, lines=True)
    aug_df = pd.read_csv(aug_data)
    train["bpps_sum"] = read_bpps_sum(train)
    train["bpps_max"] = read_bpps_max(train)
    train["bpps_nb"] = read_bpps_nb(train)
    train["bpps_v"] = read_bpps_v(train)
    train["bpps_m"] = read_bpps_m(train)

    kmeans_model = KMeans(n_clusters=200, random_state=110).fit(
        preprocess_inputs(train)[:, :, 0]
    )

    train["cluster_id"] = kmeans_model.labels_
    train = augmentation(train, aug_df)
    val_df, val_preds, model_losses = run(train, 5, 10, debug)

    for x in model_losses:
        print(*x, sep=" ")

    preds_ls = []
    for df, preds in zip(val_df, val_preds):
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
            single_df = pd.DataFrame(single_pred, columns=pred_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            single_df["SN_filter"] = df[df["id"] == uid].SN_filter.values[0]
            preds_ls.append(single_df)
    holdouts_df = pd.concat(preds_ls).groupby("id_seqpos".mean().reset_index())

    print_mse(holdouts_df)
    print_mse(holdouts_df[holdouts_df.SN_filter == 1])

