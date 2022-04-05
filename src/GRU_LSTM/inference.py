import pandas as pd
from src.GRU_LSTM.data_processing import *
from src.GRU_LSTM.model_utility import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
allocate_gpu_memory()
Ver = "GRU_LSTM_10_"
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


def predict(test, type=0, FOLD_N=5):
    print("Inferencing Model ", type, ":")
    public_df = test.query("seq_length == 107").copy()
    private_df = test.query("seq_length == 130").copy()

    public_inputs = preprocess_inputs(public_df)
    private_inputs = preprocess_inputs(private_df)

    for cv, val in enumerate(range(FOLD_N)):
        print("Fold ", cv + 1, "/", FOLD_N, "...")
        model_short = build_model(seq_len=107, pred_len=107, type=type)
        model_long = build_model(seq_len=130, pred_len=130, type=type)

        model_short.load_weights(f"./model{Ver}{type}_cv{cv}.h5")
        model_long.load_weights(f"./model{Ver}{type}_cv{cv}.h5")

        if cv == 0:
            public_preds = model_short.predict(public_inputs) / FOLD_N
            private_preds = model_long.predict(private_inputs) / FOLD_N
        else:
            public_preds += model_short.predict(public_inputs) / FOLD_N
            private_preds += model_long.predict(private_inputs) / FOLD_N
    return public_df, public_preds, private_df, private_preds


def __main__():
    test_data = "./test.json"
    aug_data = "./data/augmented_data.csv"
    test = pd.read_json(test_data, lines=True)
    debug = False
    FOLD_N = 10
    aug_df = pd.read_csv(aug_data)
    test = augmentation(test, aug_df)

    test_df, test_preds = [], []
    if debug:
        nmodel = 1
    else:
        nmodel = 5
    for i in range(nmodel):
        public_df, public_preds, private_df, private_preds = predict(test, i, FOLD_N)
        test_df.append(public_df)
        test_df.append(private_df)
        test_preds.append(public_preds)
        test_preds.append(private_preds)

    preds_ls = []
    for df, preds in zip(test_df, test_preds):
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
            single_df = pd.DataFrame(single_pred, columns=pred_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds_ls.append(single_df)
    preds_df = pd.concat(preds_ls).groupby("id_seqpos").mean().reset_index()

    submission = preds_df[
        ["id_seqpos", "reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    ]
    submission.to_csv(f"result/submission_GRU_LSTM.csv", index=False)
    print(f"wrote to submission file")

