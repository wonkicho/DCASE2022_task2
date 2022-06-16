########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import itertools
import csv
import re

# additional
import sklearn
import scipy
import numpy as np
import torch
import torchaudio
import librosa
import librosa.core
import librosa.feature
import yaml
import joblib
from tqdm import tqdm

import torch
from torch.autograd import Variable
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

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_model(model, model_dir, machine_type, states):
    """
    Save PyTorch model.
    """

    model_file_path = "{model}/model_{machine_type}.pth".format(
        model=model_dir, machine_type=machine_type
    )
    # if os.path.exists(model_file_path):
    #     print("Model already exists!")
    #     continue

    torch.save(model.state_dict(), model_file_path)
    print("save_model -> %s" % (model_file_path))

#########################################################
#train
def training(model, data_loader, criterion ,optimizer, DEVICE, epoch, config ,scheduler=None):
    """
    Perform training
    """
    model.train()  # training mode
    train_loss = 0.0
    dataset_size = 0.0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), ncols=100)
    for idx, (wav, mel, label) in pbar:
        wav = wav.float().unsqueeze(1).to(DEVICE)
        mel = mel.float().to(DEVICE)
        label = label.long().to(DEVICE)
    
        print(wav.shape, mel.shape, label.shape)

        batch_size = wav.size(0)

        preds , _= model(wav, mel, label)
        #loss = criterion(preds, label)

        dice = np.random.random() < config["fit"]["mixup"]
        if dice:
            wav, mel, label_a, label_b, lam = mixup_data(wav, mel, label, alpha = 0.3)
            wav, mel, label_a, label_b = map(Variable, (wav, mel, label_a, label_b))
            

        if dice:
            loss = mixup_criterion(criterion, preds, label_a.float(), label_b.float(), lam)
        else:
            loss = criterion(preds, label)
        
        loss.backward()  # backpropagation
        

        if (idx + 1) % config["fit"]['n_accumulate'] == 0:
            optimizer.step()  # update paramerters
            optimizer.zero_grad()  # reset gradient

            if scheduler is not None:
                scheduler.step()  # update learning rate

        train_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = train_loss / dataset_size
        pbar.set_description(f'Epoch:{epoch}'
                                     f'\tLclf:{epoch_loss:.5f}\t')


    return epoch_loss


def evaluation(model, target_dir, unique_section_names, machine_type, sec, sec_dict, criterion, CONFIG, DEVICE, mode):
    performance = []
    performance_recon = []
    total_loss = 0.0
    dataset_size = 0.0

    for section_idx, section_name in enumerate(unique_section_names):
        test_files, y_true = file_list_generator(
                        target_dir=target_dir,
                        section_name=section_name,
                        dir_name="test",
                        mode=mode,
                    )
        dataset_size += len(y_true)
        #['A1', 'A2', 'E1', 'E2', 'C2', 'C1', '40V', '28V', '31V', '37V', 1, 2]
        y_pred = [0. for _ in test_files]
        for file_idx, (file_path, label) in enumerate(zip(test_files,y_true)):
            fp = os.path.split(file_path)[1]
            fp = os.path.join(machine_type, "test", fp)
            label = []
            for i in range(len(sec)):
                if i == sec.index(sec_dict[fp]):
                    label.append(1)
                else:
                    label.append(0)
            label = np.array(label)

            x_wav, x_mel, label = test_transform(file_path, label, CONFIG, DEVICE)
            with torch.no_grad():
                model.eval()
                predict_ids, feature = model(x_wav, x_mel, label)

                label = label.float()
                loss = criterion(predict_ids, label.unsqueeze(0))
            
            total_loss += loss.item()
            probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
            label = label.cpu()
            y_pred[file_idx] = probs[torch.argmax(label)]

        if sum(np.isnan(np.array(y_true))) > 0:
            y_true = np.nan_to_num(np.array(y_true))

        if sum(np.isnan(np.array(y_pred))) > 0:
            y_pred = np.nan_to_num(np.array(y_pred))
        

        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
        p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=CONFIG["max_fpr"])
        performance.append([auc, p_auc])

    averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
    mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
    val_loss = total_loss / dataset_size

    

    return val_loss, mean_auc, mean_p_auc



class Generator(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


def test_transform(file_path, label, config, DEVICE):
    label = torch.from_numpy(np.array(label)).long().to(DEVICE)
    (x, _) = librosa.core.load(file_path, sr=config["feature"]["sr"], mono=True)

    x_wav = x[None, None, :config["feature"]["sr"] * 10]  # (1, audio_length)
    x_wav = torch.from_numpy(x_wav)
    x_wav = x_wav.float().to(DEVICE)

    x_mel = x[:config["feature"]["sr"] * 10]  # (1, audio_length)
    x_mel = torch.from_numpy(x_mel)
    x_mel = Generator(config["feature"]["sr"],
                        n_fft=config["feature"]["n_fft"],
                        n_mels=config["feature"]["n_mels"],
                        win_length=config["feature"]["win_length"],
                        hop_length=config["feature"]["hop_length"],
                        power=config["feature"]["power"],
                        )(x_mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return x_wav, x_mel, label

def fit_gamma_dist(model, target_dir, machine_type, sec, sec_dict,CONFIG, DEVICE, mode):
    """
    - Calculate anomaly scores over sections.
    - Fit gamma distribution for anomaly scores.
    - Save the parameters of the distribution.
    """

    section_names = get_section_names(target_dir, dir_name="train")
    dataset_scores = np.array([], dtype=np.float64)

    # calculate anomaly scores over sections
    for section_index, section_name in enumerate(section_names):
        section_files, _ = file_list_generator(
            target_dir=target_dir,
            section_name=section_name,
            dir_name="train",
            mode=mode,
        )
        section_scores = [0.0 for k in section_files]
        for file_idx, file_path in enumerate(section_files):
            fp = os.path.split(file_path)[1]
            fp = os.path.join(machine_type, "train", fp)
            #label = sec.index(sec_dict[fp])
            
            label = []
            for i in range(len(sec)):
                if i == sec.index(sec_dict[fp]):
                    label.append(1)
                else:
                    label.append(0)
            label = np.array(label)

            x_wav, x_mel, label = test_transform(file_path, label, CONFIG, DEVICE)
            with torch.no_grad():
                model.eval()
                predict_ids, feature = model(x_wav, x_mel, label)

            probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
            label = label.cpu()
            section_scores[file_idx] = probs[torch.argmax(label)]

            
           

        section_scores = np.array(section_scores)
        dataset_scores = np.append(dataset_scores, section_scores)

    dataset_scores = np.array(dataset_scores)

    gamma_params = scipy.stats.gamma.fit(dataset_scores)
    gamma_params = list(gamma_params)

    # save the parameters of the distribution
    score_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=CONFIG["model_directory"], machine_type=os.path.split(target_dir)[1]
    )
    joblib.dump(gamma_params, score_file_path)


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def mixup_data(x_wav,x_spec, y, alpha=0.3, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_wav.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x_wav = lam * x_wav + (1 - lam) * x_wav[index, :]
    mixed_x_spec = lam * x_spec + (1 - lam) * x_spec[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_wav, mixed_x_spec, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)