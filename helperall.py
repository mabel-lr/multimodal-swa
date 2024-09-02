import re
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch as t
import hparams
import os
import random
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import matplotlib
import matplotlib.pyplot as plt
import paths
import time

t.manual_seed(42)

labels_path = paths.labels

tokenizer = AutoTokenizer.from_pretrained(hparams.bert_model)
model = AutoModel.from_pretrained(hparams.bert_model)

def read_transcript(file_path):
  transcript = []
  with open(file_path, 'r') as file:
    for line in file:
      match = re.match(r'(\d+:\d+:\d+\.\d+) - (\d+:\d+:\d+\.\d+): (.*) \(Similarity: (\d+)\)', line)
      if match:
        start_time, end_time, text, similarity = match.groups()
        entry = {"text": text.strip(), "timestamp": f"{start_time}-{end_time}"}
        if entry:
            transcript.append(entry)
    return transcript

def add_empty_interval(transcript):
    new_transcript = []

    for i, item in enumerate(transcript):
        new_transcript.append(item)

        if i < len(transcript) - 1:
            current_end_time = item["timestamp"][1]
            # print(current_end_time)
            next_start_time = transcript[i + 1]["timestamp"][0]
            # print(next_start_time)

            if current_end_time < next_start_time:
                empty_interval = {"text": "", "timestamp": (current_end_time, next_start_time)}
                new_transcript.append(empty_interval)

    return new_transcript


def convert_transcript_timestamps(transcript):
    for item in transcript:
        time_interval_str = item["timestamp"]
        start_time_str, end_time_str = time_interval_str.split("-")
        # Convert string to datetime objects
        start_time = datetime.strptime(start_time_str, "%H:%M:%S.%f")
        end_time = datetime.strptime(end_time_str, "%H:%M:%S.%f")
        # Extract the total seconds
        start_sec = start_time.second + start_time.minute * 60 + start_time.hour * 3600 #+ start_time.microsecond / 1e6
        end_sec = end_time.second + end_time.minute * 60 + end_time.hour * 3600 #+ end_time.microsecond / 1e6
        item["timestamp"] = (start_sec, end_sec)


def zero_pad(input_array, win_len):
    array_len = len(input_array)
    zero_pad_len = 0
    max_len = (array_len // win_len + 1) * win_len  # Calculate the maximum length

    if array_len < max_len:
        zero_pad_len = max_len - array_len
    return zero_pad_len

# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    if text == '':
        output_np = np.zeros((768,), dtype=np.float32)
        output = t.from_numpy(output_np)
    else:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens = True)
        outputs = model(**inputs)
        # output = outputs.last_hidden_state
        output = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return output

def print_log(text, file_name = 'Log.txt', ends_with = '\n', display = True):
    '''
    Prints output to the log file.
    
    text        : string or List               
                        Output text

    file_name   : string
                        Target log file

    ends_with   : string
                        Ending condition for print func.

    display     : Bool
                        Whether print to screen or not.
    '''
    
    if display:
        print(text, end = ends_with)

    with open(file_name, "a") as text_file:
        print(text, end = ends_with, file = text_file)


def split_dataset(log_file, results_file,
                  bad_ops = [], random_seed = -1, test_split = 0.2, ignore_ops = False, force_test=False):
    '''
    Stplits dataset into train/valid/test sets
    with op-wise stratified sampling
    
    dataset_path    : string 
                        dataset path

    log_file        : string
                        path to save log file
    
    results_file    : string
                        path to save figure                    

    bad_ops         : list of strings
                        ops to exclude in partition

    random_seed     : int
                        random seed. If zero, dataset is not shuffled

    test_split      : float
                        portion of the test set, 
                        also valid set in train set
    '''
    ## Read dataset
    ops = [f"OP_{str(i).zfill(3)}" for i in range(1, 41)]
    ops_remove = ['OP_018', 'OP_020', 'OP_021']
    ops = [op for op in ops if op not in ops_remove]
    if len(bad_ops) != 0:
        ops = [op for op in ops if op not in bad_ops]
    if random_seed != -1:
        random.seed(random_seed)
        random.shuffle(ops)


    ## Init variables
    trainset = dict()
    validset = dict()
    testset = dict()

    train_phase_bins = np.zeros((8, ))
    valid_phase_bins = np.zeros((8, ))
    test_phase_bins = np.zeros((8, ))

    trainset_size = int(len(ops) * (1 - 2 * test_split))
    validset_size = int(len(ops) * test_split) + 1
    testset_size = len(ops) - trainset_size - validset_size

    trainset_size = 20
    validset_size = 5
    testset_size = 5

    valids = ['OP_023', 'OP_031', 'OP_029', 'OP_004']
    tests = ['OP_002', 'OP_029', 'OP_012', 'OP_014', 'OP_004']


    ## Count durations of phases in each op
    ops_dict = dict()

    total_phase_bins = np.zeros((8, )) # reference
    for i, op in enumerate(ops):
        y = t.load(labels_path + op + "/labels")
        phase_count = t.bincount(y.int().squeeze())[:-1] # exclude transition phase
        phase_count = phase_count.squeeze().clone().cpu().numpy()
        ops_dict[op] = phase_count
        
        total_phase_bins += phase_count
        

    ## Distribute ops
    H_ops = np.zeros((len(ops_dict.keys()), ))
    for i, op in enumerate(ops_dict):
        H_ops[i] = entropy(ops_dict[op], base = 2)

    for i, (h, op) in enumerate(sorted(zip(H_ops, ops_dict.keys()), reverse = True)):
        if not ignore_ops and not force_test:
            if i % 3 == 0 and len(testset.keys()) < testset_size:
                testset[op] = ops_dict[op]
                test_phase_bins += ops_dict[op]
            elif i % 3 == 1 and len(validset.keys()) < validset_size:
                validset[op] = ops_dict[op]
                valid_phase_bins += ops_dict[op]
            else:
                trainset[op] = ops_dict[op]
                train_phase_bins += ops_dict[op]
        if ignore_ops:
            if op in ['OP_015', 'OP_017']:
                if len(validset.keys()) < validset_size:
                    validset[op] = ops_dict[op]
                    valid_phase_bins += ops_dict[op]
                elif len(testset.keys()) < testset_size:
                    testset[op] = ops_dict[op]
                    test_phase_bins += ops_dict[op]
            elif len(trainset.keys()) < trainset_size:
                trainset[op] = ops_dict[op]
                train_phase_bins += ops_dict[op]
            elif len(validset.keys()) < validset_size:
                validset[op] = ops_dict[op]
                valid_phase_bins += ops_dict[op]
            elif len(testset.keys()) < testset_size:
                testset[op] = ops_dict[op]
                test_phase_bins += ops_dict[op]
        if force_test:
            if op in tests:
                testset[op] = ops_dict[op]
                test_phase_bins += ops_dict[op]
            elif op in valids:
                validset[op] = ops_dict[op]
                valid_phase_bins += ops_dict[op]
            else:
                trainset[op] = ops_dict[op]
                train_phase_bins += ops_dict[op]

    assert np.array_equal(total_phase_bins, train_phase_bins + valid_phase_bins + test_phase_bins)
    

    print_log("\tTrainset [{}] OPs\t: {}".format(len(trainset.keys()),trainset.keys()), log_file)
    print_log("\tValidset [{}] OPs\t: {}".format(len(validset.keys()), validset.keys()), log_file)
    print_log("\tTestset [{}] OPs\t\t: {}".format(len(testset.keys()), testset.keys()), log_file)

    print_log("\tTrainset Entropy\t: {}".format(entropy(train_phase_bins)), log_file)
    print_log("\tValidset Entropy\t: {}".format(entropy(valid_phase_bins)), log_file)
    print_log("\tTestset Entropy\t: {}".format(entropy(test_phase_bins)), log_file)


    phases = ['Preparation', 'Puncture', 'GuideWire', 'CathPlacement', 
        'CathPositioning', 'CathAdjustment', 'CathControl', 'Closing']

    matplotlib.rc('font',family='Times New Roman')
    
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.linspace(0, 1, 9))

    fig, axs = plt.subplots(3, 1, sharex = True, figsize = (16, 12), dpi = 600)

    axs[0].bar(range(8), train_phase_bins / total_phase_bins, color = color_list)
    axs[0].set_title("Training Set [{} OPs]".format(trainset_size), fontsize = 26)
    axs[0].set_yticks(np.linspace(0, 1, num = 5), labels = np.linspace(0, 1, num = 5) ,fontsize = 20)
    axs[0].margins(x = 0.01)

    axs[1].bar(range(8), valid_phase_bins / total_phase_bins, color = color_list)
    axs[1].set_title("Validation Set [{} OPs]".format(validset_size), fontsize = 26)
    axs[1].set_yticks(np.linspace(0, 2*(validset_size/len(ops)), num = 5), labels = np.linspace(0, 1, num = 5) ,fontsize = 20)
    axs[1].margins(x = 0.01)

    axs[2].set_title("Test Set [{} OPs]".format(testset_size), fontsize = 26)
    axs[2].bar(range(8), test_phase_bins / total_phase_bins, color = color_list)
    axs[2].set_yticks(np.linspace(0, 2*(testset_size/len(ops)), num = 5), labels = np.linspace(0, 1, num = 5) ,fontsize = 20)
    axs[2].set_xticks(np.arange(len(phases)), labels = phases, fontsize = 20, rotation = 15)
    axs[2].margins(x = 0.01)

    plt.savefig(results_file + "/class_dist.png", bbox_inches = 'tight')
    plt.close('all')

    return list(trainset.keys()), list(validset.keys()), list(testset.keys())


def plot_loss(train_loss, valid_loss, save_path):

    train_loss = train_loss[train_loss != 0] # cut tailing zeros after early stopper callback
    valid_loss = valid_loss[valid_loss != 0]

    plt.figure(dpi = 600, constrained_layout = True) 
    plt.style.use('fivethirtyeight')
    plt.plot(train_loss,  linewidth = 2, label = 'Train Loss')
    plt.plot(valid_loss,  linewidth = 2, label = 'Valid Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Loss Functions')
    plt.xlabel('Epochs')
    plt.savefig(save_path + '/loss.png')
    plt.close('all')

def plot_ribbon(data, title, out_path, repeat = 1024):
    ''' Plots color ribbon with legend
    
    data        : np.array [1xN]
                    Data to plot

    title       : str
                    Title and save name of the figure, e.g. OP name

    out_path    : str
                    path to save.

    repeat      : int
                    Vertical width of the ribbon 
    '''

    phases = ['Preparation','Puncture', 'GuideWire', 'CathPlacement', 
        'CathPositioning', 'CathAdjustment', 'CathControl', 'Closing']

    # ensure data type
    assert type(data) == type(np.zeros([1, 1])), "Input data should be a numpy array"

    # ensure horizontal
    if data.shape[1] == 1:
        data = np.transpose(data)
    
    data = np.repeat(data, repeats = repeat, axis = 0)

    formatter = matplotlib.ticker.FuncFormatter(lambda s, 
        x: time.strftime('%M:%S', time.gmtime(s // 60)))
    xtick_pos = np.linspace(0, data.shape[1], data.shape[1] // 350)

    matplotlib.rc('font',family='Times New Roman')

    # discrete cmap
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.linspace(0, 1, 8))
    disc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('DMap', color_list, 8)

    # plot
    plt.figure(dpi = 600, figsize = (28,12))
    plt.matshow(data, cmap = disc_cmap)
    plt.grid(False)
    plt.yticks([])
    cbar = plt.colorbar(ticks = range(len(phases)))
    cbar.ax.set_yticks(np.arange(len(phases)), labels = phases, fontsize = 18)
    plt.gca().xaxis.tick_bottom()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(xtick_pos, fontsize = 18)
    plt.xlabel('Time (HH:MM)', fontsize = 18)
    plt.title(title, fontsize = 24, pad = 10)
    plt.savefig(out_path + title + ".png", bbox_inches = 'tight')
    plt.close('all')

def plot_confusion(y_estim, y_ground, title, out_path):
    conf_mat = confusion_matrix(y_ground, y_estim)
    plt.figure(dpi = 600)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot()
    plt.title(title)
    plt.grid(False)
    plt.savefig(out_path + title + "_CM.pdf", bbox_inches = 'tight', dpi = 600)
    plt.close('all')

def postprocessing(arr, threshold=10):
    result = arr.copy()

    for i, item in enumerate(result):
        # Check the range around the current index
        start_index = max(0, i - threshold//2)
        end_index = min(len(arr), i + threshold//2 + 1)
        subarray = result[start_index:end_index]

        different = [x for x in subarray if x != item]

        # Check if there are other '1's within the range
        if np.sum(subarray == item) < len(different):
            result[i] = abs(item-1)
        else:
            result[i] = item

    return result



def plot_comparison(pred, label, path, op):
    out_path = path + op
    fig,  (ax1, ax3)= plt.subplots(2,1, figsize=(15, 6)) #gridspec_kw={'width_ratios': [1, 1, 0.05]}

    # Plotting ground truth tensor
    if len(label.unique())<8:
        phases = ['2: Puncture', '3: GuideWire', '4: CathPlacement', 
            '5: CathPositioning', '6: CathAdjustment', '7: CathControl','Transition']
    else:
        phases = ['1: Preparation', '2: Puncture', '3: GuideWire', '4: CathPlacement', 
            '5: CathPositioning', '6: CathAdjustment', '7: CathControl', '8: Closing','Transition']

    phases1 = phases
    data = np.expand_dims(label.squeeze(1), 1)
    repeat = 512
    if data.shape[1] == 1:
        data = np.transpose(data)

    data = np.repeat(data, repeats=repeat, axis=0)

    # Discrete colormap
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.arange(len(phases))) # np.linspace(0,1, len(phases))
    disc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('DMap', color_list, len(phases))

    # Plot
    matshow_obj1 = ax1.matshow(data, cmap=disc_cmap, aspect='auto')
    ax1.grid(False)

    # Hide x-axis and y-axis ticks and labels
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(f'True and estimated Surgical Phases in {op}', fontsize= 16)


    # Plotting label tensor
    if len(label.unique())<8:
        phases = ['2: Puncture', '3: GuideWire', '4: CathPlacement', 
            '5: CathPositioning', '6: CathAdjustment', '7: CathControl']
    else:
        phases = ['1: Preparation', '2: Puncture', '3: GuideWire', '4: CathPlacement', 
            '5: CathPositioning', '6: CathAdjustment', '7: CathControl', '8: Closing']

    data = np.expand_dims(pred, 1)
    repeat = 512
    if data.shape[1] == 1:
        data = np.transpose(data)

    data = np.repeat(data, repeats=repeat, axis=0)

    # Discrete colormap
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.arange(len(phases)))
    disc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('DMap', color_list, len(phases))

    # Plot
    matshow_obj2 = ax3.matshow(data, cmap=disc_cmap, aspect='auto')
    ax3.grid(False)

    # Hide x-axis and y-axis ticks and labels
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    color_bar = fig.colorbar(matshow_obj1, cax=cbar_ax, ticks=range(len(phases1)))

    # Set the y tick labels
    color_bar.set_ticks(range(len(phases1)))
    color_bar.set_ticklabels(phases1)

    fig.text(0.1, 0.7, 'Ground\nTruth', va='center', ha='center')
    fig.text(0.1, 0.3, 'Estimated\nPhases', va='center', ha='center')


    plt.rcParams["font.family"] = "Times New Roman"
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.savefig(out_path +'_comp.svg', bbox_inches = 'tight', format='svg')
