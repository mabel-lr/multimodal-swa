
## Data
test_split = 0.15
random_seed = 2220
bert_model = "dbmdz/bert-base-german-uncased"   #"GerMedBERT/medbert-512"

## Training
batch_size = 180 # 180
learning_rate = 9e-6 #e-6
weight_decay = 1e-6
epochs = 100
num_stages = 2
num_layers_speech = 4  # TCN
num_layers_x = 7
nlayers = 4 # LSTM
num_f_maps_speech = 256
num_f_maps_x = 304
dim = 1024
dim_x = 1216
compute_dim_speech = 256
compute_dim_x = 304
wav2vec_dim = 50 # 149 w3
mfcc_dim = 80
mfcc_dim2 = 40
tr_dim = 768
num_classes_x = 8
num_classes_speech = 3
causal_conv = False
num_resblocks = 2
patience_lim = 12    # early stopper callback patience
escb_beta = 0.25    # sensitivity margin
