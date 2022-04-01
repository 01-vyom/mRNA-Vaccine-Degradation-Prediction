import numpy as np
import pandas as pd
import tensorflow.keras.layers as L
import keras.backend as K
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}


def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices("GPU")

    if physical_devices:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.set_visible_devices(physical_devices[gpu_number], "GPU")
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")


def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)


def mcrmse(y_actual, y_pred, num_scored=len(pred_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=256, type=1
):
    inputs = L.Input(shape=(seq_len, 8))

    # split categorical and numerical features and concatenate them later.
    categorical_feat_dim = 3
    categorical_fea = inputs[:, :, :categorical_feat_dim]
    numerical_fea = inputs[:, :, 3:]

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1], embed.shape[2] * embed.shape[3])
    )
    reshaped = L.concatenate([reshaped, numerical_fea], axis=2)

    if type == 0:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    elif type == 1:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    elif type == 2:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    elif type == 3:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif type == 4:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)
    return model


def print_mse(prd):
    val = pd.read_json("./train.json", lines=True)

    val_data = []
    for mol_id in val["id"].unique():
        sample_data = val.loc[val["id"] == mol_id]
        sample_seq_length = sample_data.seq_length.values[0]
        for i in range(68):
            sample_dict = {
                "id_seqpos": sample_data["id"].values[0] + "_" + str(i),
                "reactivity_gt": sample_data["reactivity"].values[0][i],
                "deg_Mg_pH10_gt": sample_data["deg_Mg_pH10"].values[0][i],
                "deg_Mg_50C_gt": sample_data["deg_Mg_50C"].values[0][i],
            }
            val_data.append(sample_dict)
    val_data = pd.DataFrame(val_data)
    val_data = val_data.merge(prd, on="id_seqpos")

    rmses = []
    mses = []
    for col in ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]:
        rmse = ((val_data[col] - val_data[col + "_gt"]) ** 2).mean() ** 0.5
        mse = ((val_data[col] - val_data[col + "_gt"]) ** 2).mean()
        rmses.append(rmse)
        mses.append(mse)
        print(col, rmse, mse)
    print(np.mean(rmses), np.mean(mses))
