########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import itertools
import re

# additional
import sklearn
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
from tqdm import tqdm

import torch
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")

    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2022 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.dev:
        flag = True
    elif args.eval:
        flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load(yaml_path):
    with open(yaml_path) as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file input
def file_load(wav_name, mono=False):
    """
    load .wav file.
    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data
    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors


########################################################################


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(config, mode):
    """
    param : dict
        baseline.yaml data
    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=config["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=config["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# get machine IDs
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files
    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        dir_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files
    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # development
    if mode:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_normal,
                                                                                                     ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     prefix_normal=prefix_anomaly,
                                                                                                     ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                                     dir_name=dir_name,
                                                                                                     section_name=section_name,
                                                                                                     ext=ext))
        files = sorted(glob.glob(query))
        labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################

def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data


def get_dataloader(dataset, CONFIG):
    """
    Make dataloader from dataset for training.
    """
    train_size = int(len(dataset) * (1.0 - CONFIG["fit"]["validation_split"]))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG["fit"]["batch_size"],
        shuffle=CONFIG["fit"]["shuffle"],
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CONFIG["fit"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # dataloader of training data for evaluation only
    data_loader_eval_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CONFIG["fit"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    return data_loader_train, data_loader_val, data_loader_eval_train

#########################################################
#train
def training(model, data_loader, criterion ,optimizer, DEVICE, epoch, scheduler=None):
    """
    Perform training
    """
    model.train()  # training mode
    train_loss = 0.0
    pbar = tqdm(data_loader, total=len(data_loader), ncols=100)
    for idx, (wav, mel, label) in enumerate(pbar):
        wav = wav.float().unsqueeze(1).to(DEVICE)
        mel = mel.float().to(DEVICE)
        label = label.long().to(DEVICE)
        #label = label.softmax(dim=1)
        
        preds , _= model(wav, mel, label)
        
        optimizer.zero_grad()  # reset gradient
        loss = criterion(preds, label)
        pbar.set_description(f'Epoch:{epoch}'
                                     f'\tLclf:{loss.item():.5f}\t')
        loss.backward()  # backpropagation
        train_loss += loss.item()
        optimizer.step()  # update paramerters

    if scheduler is not None:
        scheduler.step()  # update learning rate

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for wav, mel, label in data_loader:
            wav = wav.float().unsqueeze(1).to(DEVICE)
            mel = mel.float().to(DEVICE)
            label = label.long().to(DEVICE)
            outputs,features = model(wav, mel, label)
            probs = - torch.log_softmax(outputs, dim=1).mean(dim=0).squeeze().cpu().numpy()
            y_pred.append(probs)
            y_true.append(label.cpu().numpy())
            #correct += (predicted == label).sum().item()
            #total += label.size(0)
    
    
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    print(auc)
    #accuracy = 100 * float(correct / total)
    #print(
    #    "accuracy: {:.6f}% ({}/{})".format(
    #      accuracy , correct, total
    #    ),
    #)

    return train_loss

def validation(model, data_loader, criterion ,DEVICE , CONFIG):
    """
    Perform validation
    """
    model.eval()  # validation mode
    val_loss = 0.0
    correct = 0
    total = 0
    y_pred = [0. for _ in len(data_loader)]
    pbar = tqdm(data_loader, total=len(data_loader), ncols=100)
    with torch.no_grad():
        for idx, (wav, mel, label) in enumerate(pbar):
            wav = wav.float().unsqueeze(1).to(DEVICE)
            mel = mel.float().to(DEVICE)
            label = label.long().to(DEVICE)
            
            preds, features = model(wav, mel, label)
            loss = criterion(preds, label)
            val_loss += loss.item()

            _, predicted = torch.max(preds, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
            
            probs = - torch.log_softmax(preds, dim=1).mean(dim=0).squeeze().cpu().numpy()
            y_pred[idx] = probs[label]
            
    auc = sklearn.metrics.roc_auc_score(label, y_pred)
    p_auc = sklearn.metrics.roc_auc_score(label, y_pred, max_fpr=CONFIG["max_fpr"])
    
      
    print("loss: {:.6f} - ".format(val_loss / len(data_loader)), end="")
    print(
        "accuracy: {:.6f}% ({}/{})".format(
            100 * float(correct / total), correct, total
        ),
    )
    
    print(auc, p_auc)

def create_test_file_list(target_dir,
                          id_name,
                          dir_name='test',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels