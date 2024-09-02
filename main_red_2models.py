from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from datetime import datetime
import torch as t
from mymodel_speech import M3X1 as M3X1S1
from mymodel_xray import M3X1 as M3X1X

import torch.nn.functional as F
import torch.nn as nn
import hparams_all as hp
import helperall as h
import numpy as np
import shutil
import sys
import os
import paths
from torch.utils.tensorboard import SummaryWriter

labels_path = paths.labels
xray_path = paths.xray
audio_path = paths.audio
mfcc_path = paths.mfcc

TRAINING_LABEL = "Test20"

## Start
log_path = '/home/belen/MasterThesisSLP/indpmodels/allphases/Tests/NewTests/' + TRAINING_LABEL +'/'
log_file = log_path + "log.txt"


if os.path.exists(log_path):
    sys.exit("This training label already exists :( \n\"{}\""
             .format(TRAINING_LABEL))
else:
    os.mkdir(log_path)
    os.mkdir(log_path + "/code")
    os.mkdir(log_path + "/results")

homedir = ""
for f in os.listdir(homedir):
    if f.endswith(".py"):
        shutil.copyfile(homedir+f, log_path + "/code/" + f)

writer = SummaryWriter(log_dir=log_path + '/runs')

h.print_log('\nDescription: \n', log_file)

h.print_log('\n\tTesting "{}" started at: {} \n'.format(TRAINING_LABEL,
            datetime.now().strftime('%d-%m-%Y %H:%M:%S')), log_file)
h.print_log('GPU: {}'.format(t.cuda.get_device_name()), log_file)
h.print_log('Properties: {}'.format(t.cuda.get_device_properties("cuda")),
            log_file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


## Load data
trainset, validset, testset = h.split_dataset(log_file = log_file,
                                              results_file = log_path + '/results',
                                              bad_ops = ['OP_022', 'OP_028', 
                                               'OP_037', 'OP_040', 'OP_036', 'OP_009', 'OP_011', 'OP_015'],
                                              random_seed = hp.random_seed,
                                              test_split = hp.test_split)


   
class OPDataset(Dataset):
    def __init__(self, op):
        self.data = self.get_data(op)
        self.op = op

    def __getitem__(self,idx):
        x = self.data[0][idx].clone().detach().float().to(device)
        p = self.data[1][idx].clone().detach().float().to(device)
        a = self.data[2][idx].clone().detach().float().to(device)
        g = self.data[3][idx].clone().detach().float().to(device)
        l = self.data[4][idx].clone().detach().float().to(device)
        y = self.data[5][idx].clone().detach().long().to(device)
        return [x, p, a, g, l, y]
    
    def __len__(self):
        return self.data[5].shape[0]

    def get_data(self, op):
        xray = t.load(xray_path + op + "/x_ray")            # tensor created with create_dataset.py
        p_m = t.load(audio_path + op + "/physician_mic")
        a_m = t.load(audio_path + op + "/assistant_mic")
        g_m = t.load(mfcc_path + op + "/mfcc")
        log = t.load(audio_path + op + '/log')
        label = t.load(labels_path + op + "/labels")
        return [xray, p_m, a_m, g_m, log, label]




## Test
h.print_log("\n\n\ttesting...\n\n", log_file)

test_metrics = np.zeros((len(testset), 5)) # [acc, recall, precision, f1, jaccard]
# Load the checkpoint
checkpoint_speech1 = t.load(paths.models + 'speech.ckp')
best_model_speech1 = M3X1S1().to(device)  # Replace with your model class
best_model_speech1.load_state_dict(checkpoint_speech1['state_dict'])
best_model_speech1.eval()

checkpoint_xray = t.load(paths.model_xray + 'xray.ckp')
best_model_xray = M3X1X().to(device)  # Replace with your model class
best_model_xray.load_state_dict(checkpoint_xray['state_dict'])
best_model_xray.eval()

phases_speech1 = [0,1]


for i_op, op in enumerate(testset):

    op_dataset = OPDataset(op)
    op_dataloader = DataLoader(dataset = op_dataset, 
                            batch_size = hp.batch_size,
                            shuffle = False)

    y_estim = t.zeros(op_dataset.__len__(),)
    y_ground = t.zeros(op_dataset.__len__(),)
    start_idx = 0
    end_idx = hp.batch_size
    y_last = y_estim[start_idx:end_idx]

    y_prev = t.zeros((1, hp.compute_dim), device = device) #(1, 128)
    j = 0
    k = 0
    for i, (x, s, a, g, l, y) in enumerate(op_dataloader):
        with t.no_grad():
            if any(item in phases_speech1 for item in y_last) and not t.any(y_last == 2) and not ((y_last == 1).sum().item() > (y_last == 0).sum().item()): # not (y_last.unique().numpy() == 1).all():
                y_hat = best_model_speech1(s, a, g, y_prev.clone().detach())
                y_estim[start_idx:end_idx] = t.argmax(y_hat[:,:,-1].clone().detach(), dim = 1)
                y_last = y_estim[start_idx:end_idx]
            else:
                j+=1
                if j == 1:
                    y_prev = t.zeros((1, hp.compute_dim_x), device = device) #(1, 128)
                y_hat = best_model_xray(x, l, y_prev.clone().detach())
                y_estim[start_idx:end_idx] = t.argmax(y_hat[:,:,-1].clone().detach(), dim = 1) + 1
                y_last = y_estim[start_idx:end_idx]

                

        y_prev[0, -min(hp.compute_dim, x.shape[0]):] = t.argmax(y_hat[:,:,-1], 
                                    dim = 1)[-min(hp.compute_dim, x.shape[0]):]

        y_ground[start_idx:end_idx] = y.clone().detach().squeeze()

        prev_start = start_idx
        prev_end = end_idx
        start_idx = end_idx
        end_idx = min(end_idx + hp.batch_size, op_dataset.__len__())
    
    # scores
    y_ground = y_ground.cpu().detach().numpy()
    y_estim = y_estim.cpu().detach().numpy()
    
    transition_idx = np.where(y_ground == 8)[0] # remove transition for evaluation
    y_estim = np.delete(y_estim, transition_idx)
    y_ground = np.delete(y_ground, transition_idx)
    
    test_metrics[i_op, 0] = accuracy_score(y_ground, y_estim)
    h.print_log("\t\t{} Accuracy:\t{:.5f}".format(op, test_metrics[i_op, 0]), log_file)
    test_metrics[i_op, 1] = recall_score(y_ground, y_estim, average = "weighted")
    h.print_log("\t\t{} Recall:\t\t{:.5f}".format(op, test_metrics[i_op, 1]), log_file)
    test_metrics[i_op, 2] = precision_score(y_ground, y_estim, average = "weighted")
    h.print_log("\t\t{} Precision:\t{:.5f}".format(op, test_metrics[i_op, 2]), log_file)
    test_metrics[i_op, 3] = f1_score(y_ground, y_estim, average = 'weighted')
    h.print_log("\t\t{} F1:\t{:.5f}".format(op, test_metrics[i_op, 3]), log_file)
    test_metrics[i_op, 4] = jaccard_score(y_ground, y_estim, average = 'weighted')
    h.print_log("\t\t{} Jaccard:\t{:.5f}".format(op, test_metrics[i_op, 4]), log_file)

    h.plot_confusion(y_estim, y_ground, op, log_path + "/results/")

    # Ground truth
    label = t.load(labels_path + op + "/labels")
    h.plot_comparison(y_estim, label, log_path + "/results/", op)
    

h.print_log("\n\t\tAccuracy:\t{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 0]),
                                                              np.std(test_metrics[:, 0])), 
                                                              log_file)
h.print_log("\n\t\tRecall:\t\t{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 1]),
                                                              np.std(test_metrics[:, 1])), 
                                                              log_file)
h.print_log("\n\t\tPrecision:\t{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 2]),
                                                              np.std(test_metrics[:, 2])), 
                                                              log_file)
h.print_log("\n\t\tF1:\t\t\t{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 3]),
                                                              np.std(test_metrics[:, 3])), 
                                                              log_file)
h.print_log("\n\t\tJaccard:\t\t{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 4]),
                                                              np.std(test_metrics[:, 4])), 
                                                              log_file)


## Log finish time
h.print_log('\n\tTraining finished at: {} \n'.format(
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')), log_file)
