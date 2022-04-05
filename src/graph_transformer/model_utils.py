import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K


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


def mcrmse(t, p, seq_len_target):
    ## calculate mcrmse score by using numpy
    t = t[:, :seq_len_target]
    p = p[:, :seq_len_target]

    score = np.mean(np.sqrt(np.mean(np.mean((p - t) ** 2, axis=1), axis=0)))
    return score


def attention(x_inner, x_outer, n_factor, dropout):
    x_Q = L.Conv1D(
        n_factor,
        1,
        activation="linear",
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
    )(x_inner)
    x_K = L.Conv1D(
        n_factor,
        1,
        activation="linear",
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
    )(x_outer)
    x_V = L.Conv1D(
        n_factor,
        1,
        activation="linear",
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
    )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att


def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(
            n_factor,
            kernel_initializer="glorot_uniform",
            bias_initializer="glorot_uniform",
        )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x


def res(x, unit, kernel=3, rate=0.1):
    h = L.Conv1D(unit, kernel, 1, padding="same", activation=None)(x)
    h = L.LayerNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])


def forward(x, unit, kernel=3, rate=0.1):
    h = L.Conv1D(unit, kernel, 1, padding="same", activation=None)(x)
    h = L.LayerNormalization()(h)
    h = L.Dropout(rate)(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate)
    return h


def adj_attn(x, adj, unit, n=2, rate=0.1):
    x_a = x
    x_as = []
    for i in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a)  ## aggregate neighborhoods
        x_as.append(x_a)
    if n == 1:
        x_a = x_as[0]
    else:
        x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a


def get_base(X_node, adj_matrix):
    ## base model architecture
    ## node, adj -> middle feature

    node = tf.keras.Input(shape=(None, X_node.shape[2]), name="node")
    adj = tf.keras.Input(shape=(None, None, adj_matrix.shape[3]), name="adj")

    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis=3)([adj, adj_learned])

    xs = []
    xs.append(node)
    x1 = forward(node, 128, kernel=3, rate=0.0)
    x2 = forward(x1, 64, kernel=6, rate=0.0)
    x3 = forward(x2, 32, kernel=15, rate=0.0)
    x4 = forward(x3, 16, kernel=30, rate=0.0)
    x = L.Concatenate()([x1, x2, x3, x4])

    for unit in [64, 32]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate=0.0)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel=30)

        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)

    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs=[node, adj], outputs=[x])
    return model


def get_ae_model(base, X_node, adj_matrix):
    ## denoising auto encoder part
    ## node, adj -> middle feature -> node

    node = tf.keras.Input(shape=(None, X_node.shape[2]), name="node")
    adj = tf.keras.Input(shape=(None, None, adj_matrix.shape[3]), name="adj")

    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64, rate=0.3)
    p = L.Dense(X_node.shape[2], "sigmoid")(x)

    loss = -tf.reduce_mean(
        20 * node * tf.math.log(p + 1e-4) + (1 - node) * tf.math.log(1 - p + 1e-4)
    )
    model = tf.keras.Model(inputs=[node, adj], outputs=[loss])

    opt = get_optimizer()
    model.compile(optimizer=opt, loss=lambda t, y: y)
    return model


# loss functions
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(1))
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


def get_model(base, X_node, adj_matrix, seq_len_target):
    ## regression part
    ## node, adj -> middle feature -> prediction of targets

    node = tf.keras.Input(shape=(None, X_node.shape[2]), name="node")
    adj = tf.keras.Input(shape=(None, None, adj_matrix.shape[3]), name="adj")

    x = base([node, adj])
    x = forward(x, 128, rate=0.4)
    x = L.Dense(5, None)(x)

    model = tf.keras.Model(inputs=[node, adj], outputs=[x])

    opt = get_optimizer()

    def mcrmse_loss(t, y, seq_len_target=seq_len_target):
        ## calculate mcrmse score by using tf
        t = t[:, :seq_len_target]
        y = y[:, :seq_len_target]

        loss = tf.reduce_mean(
            tf.sqrt(tf.reduce_mean(tf.reduce_mean((t - y) ** 2, axis=1), axis=0))
        )
        return loss

    model.compile(optimizer=opt, loss=mcrmse_loss)
    return model


def get_optimizer():
    adam = tf.optimizers.Adam()
    return adam
