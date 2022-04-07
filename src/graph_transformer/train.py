import gc
import os
from matplotlib import pyplot as plt
import pandas as pd
from src.graph_transformer.data_processing import *
from src.graph_transformer.model_utils import *
import tensorflow as tf
from sklearn.model_selection import KFold

gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Code for training the auto encoder
def train_auto_encoder(
    X_node,
    X_node_pub,
    X_node_pri,
    adjacency_matrix,
    adjacency_matrix_pub,
    adjacency_matrix_pri,
    epochs,
    epochs_each,
    batch_size,
    save_path,
):
    base = get_base(X_node, adjacency_matrix)
    ae_model = get_ae_model(base, X_node, adjacency_matrix)

    # Iterate epoch //epochs_each times over the three data
    for i in range(epochs // epochs_each):
        print(f"------ {i} ------")
        # Using the training dataset
        ae_model.fit(
            [X_node, adjacency_matrix],
            [X_node[:, 0]],
            epochs=epochs_each,
            batch_size=batch_size,
        )
        gc.collect()
        print("--- public ---")
        # using the public dataset
        ae_model.fit(
            [X_node_pub, adjacency_matrix_pub],
            [X_node_pub[:, 0]],
            epochs=epochs_each,
            batch_size=batch_size,
        )
        print("--- private ---")
        # Use the private dataset
        ae_model.fit(
            [X_node_pri, adjacency_matrix_pri],
            [X_node_pri[:, 0]],
            epochs=epochs_each,
            batch_size=batch_size,
        )
        gc.collect()
    print("****** save ae model ******")

    base.save_weights(save_path)

'''
# Train the GCN model, takes as input the 
# @X_node: It is the train node features,
# @adjacency_matrix: It is the edge feature in adjacency matrix form
# @seq_len_target: It is the sequence length of the input data
# @epochs: The number of epochs to run the training
# @batch_size: The batch size to be used for the training
# @model_path: The path where the trained model is to be saved.
# @ae_model_path: Specifies the location of the ae pretrained model. 
#                  If null doesnt use any pre trained model.
# @plt_name: Name of the plot
# @n_fold: The number of folds to use for training
# @validation_frequency: Frequency in epochs after which the validation is to be run
# @y: The ground truth target
'''
def train_gcn(
    X_node,
    adjacency_matrix,
    seq_len_target,
    epochs,
    batch_size,
    model_path,
    ae_model_path=None,
    plt_name="withhout_ae",
    n_fold=5,
    validation_frequency=1,
    y=None,
):
    kfold = KFold(n_fold, shuffle=True, random_state=42)
    legends = []
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_node, adjacency_matrix)):
        gc.collect()
        tf.keras.backend.clear_session()
        print("Fold ", fold + 1, "/", n_fold, ": Training Started...")
        X_node_tr = X_node[tr_idx]
        X_node_va = X_node[va_idx]
        As_tr = adjacency_matrix[tr_idx]
        As_va = adjacency_matrix[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        base = get_base(X_node, adjacency_matrix)
        if ae_model_path:
            base.load_weights(ae_model_path)
        model = get_model(base, X_node, adjacency_matrix, seq_len_target)
        filepath_list = model_path.split(".")
        filepath = f"{filepath_list[0]}_{fold}.{filepath_list[1]}"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        history = model.fit(
            [X_node_tr, As_tr],
            [y_tr],
            validation_data=([X_node_va, As_va], [y_va]),
            epochs=epochs,
            batch_size=batch_size,
            validation_freq=validation_frequency,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(),
                model_checkpoint_callback,
            ],
        )
        print(
            f"Min training loss={min(history.history['loss'])}, min validation loss={min(history.history['val_loss'])}"
        )
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        legends.append(f"loss_fold_{fold}")
        legends.append(f"val_loss_fold_{fold}")

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(
        legends,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="x-small",
    )
    plt.savefig("./result/GCN_Model_" + plt_name + ".png", bbox_inches="tight", dpi=600)
    plt.show()


def __main__():
    denoise = True  # if True, use train data whose signal_to_noise > 1
    allocate_gpu_memory()
    path = "."
    aug_data = "./data/augmented_data.csv"
    n_fold = 7
    aug_df = pd.read_csv(aug_data)
    train = pd.read_json(f"{path}/train.json", lines=True)
    train = augmentation(train, aug_df)
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

    adjacency_matrix = get_adj_matrix(train)
    adjacency_matrix_pub = get_adj_matrix(test_pub)
    adjacency_matrix_pri = get_adj_matrix(test_pri)

    X_node = get_node_features(train)
    X_node_pub = get_node_features(test_pub)
    X_node_pri = get_node_features(test_pri)

    epochs_list = [30, 10, 3, 3, 5, 5]
    batch_size_list = [8, 16, 32, 64, 128, 256]

    # Train model without auto encoder
    epochs = epochs_list[0]
    batch_size = batch_size_list[1]

    model_path = "./model_without_ae.h5"
    train_gcn(
        X_node,
        adjacency_matrix,
        seq_len_target,
        epochs,
        batch_size,
        model_path,
        n_fold=n_fold,
        y=y,
    )

    # Train model with auto encoder
    ae_epochs = 30  # epoch of training of denoising auto encoder
    ae_epochs_each = 10  # epoch of training of denoising auto encoder each time.
    ae_batch_size = 32
    ae_path = "./base_ae"
    train_auto_encoder(
        X_node,
        X_node_pub,
        X_node_pri,
        adjacency_matrix,
        adjacency_matrix_pub,
        adjacency_matrix_pri,
        ae_epochs,
        ae_epochs_each,
        ae_batch_size,
        ae_path,
    )

    epochs_list = [30, 10, 3, 3, 5, 5]
    batch_size_list = [8, 16, 32, 64, 128, 256]

    epochs = epochs_list[0]
    batch_size = batch_size_list[1]

    model_path = "./model_with_ae.h5"
    train_gcn(
        X_node,
        adjacency_matrix,
        seq_len_target,
        epochs,
        batch_size,
        model_path,
        ae_model_path=ae_path,
        n_fold=n_fold,
        y=y,
    )

