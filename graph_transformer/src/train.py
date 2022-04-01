from sklearn.model_selection import KFold

def train_auto_encoder(X_node, adjacency_matrix, epochs, epochs_each, batch_size, save_path):
    base = get_base(X_node, adjacency_matrix)
    ae_model = get_ae_model(base, X_node, adjacency_matrix)
    for i in range(epochs//epochs_each):
        print(f"------ {i} ------")
        ae_model.fit([X_node, adjacency_matrix], [X_node[:,0]],
                  epochs = epochs_each,
                  batch_size = batch_size)
        gc.collect()
    print("****** save ae model ******")
    base.save_weights(save_path)


def train_gcn(X_node, adjacency_matrix, seq_len_target, epochs, batch_size,
              model_path, ae_model_path=None, n_fold=2, validation_frequency=1):
    kfold = KFold(n_fold, shuffle = True, random_state = 42)

    legends = []
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X_node, adjacency_matrix)):

        gc.collect()
        tf.keras.backend.clear_session()
        print(f'\nFold - {fold}\n')
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
        filepath_list = model_path.split('.')
        filepath = filepath_list[0] + f'_{fold}' + filepath_list[1]
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=filepath, save_weights_only=True,
                                    monitor='val_loss',mode='min',save_best_only=True)

        history = model.fit([X_node_tr, As_tr], [y_tr],
                            validation_data=([X_node_va, As_va], [y_va]),
                            epochs = epochs,
                            batch_size = batch_size, 
                            validation_freq = validation_frequency,
                            callbacks=[model_checkpoint_callback]
                       )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        legends.append(f'loss_fold_{fold}')
        legends.append(f'val_loss_fold_{fold}')

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(legends, loc='upper left')
    plt.show()



if __main__():
	denoise = True # if True, use train data whose signal_to_noise > 1

	path = "../input/stanfordcovidvaccine"

	train = pd.read_json(f"{path}/train.json",lines=True)
	if denoise:
	    train = train[train.signal_to_noise > 1].reset_index(drop = True)
	    
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
	y = np.stack(y_train, axis = 2)
	y.shape

	adjacency_matrix = get_adj_matrix(train)
	X_node = get_node_features(train)

	epochs_list = [30, 10, 3, 3, 5, 5]
	batch_size_list = [8, 16, 32, 64, 128, 256] 

	epochs = epochs_list[0]
	batch_size = batch_size_list[1]

	model_path = "./model_without_ae.h5"
	train_gcn(X_node, adjacency_matrix, seq_len_target, epochs, batch_size,
	              model_path)	


	ae_epochs = 20 # epoch of training of denoising auto encoder
	ae_epochs_each = 5 # epoch of training of denoising auto encoder each time.                  
	ae_batch_size = 32
	ae_path = "./base_ae"
	train_auto_encoder(X_node, adjacency_matrix, ae_epochs, ae_epochs_each, 
	                   ae_batch_size, ae_path)

	epochs_list = [30, 10, 3, 3, 5, 5]
	batch_size_list = [8, 16, 32, 64, 128, 256] 

	epochs = epochs_list[0]
	batch_size = batch_size_list[1]

	model_path = "./model_with_ae.h5"
	train_gcn(X_node, adjacency_matrix, seq_len_target, epochs, batch_size,
	              model_path, ae_model_path=ae_path)			

