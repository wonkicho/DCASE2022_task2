import datetime
import enum
import os
import sys
from itertools import chain

import joblib
import numpy
import scipy.stats
import torch
import torch.utils.data
from scipy.special import softmax
from torch import optim
from torchinfo import summary
import pandas as pd


from utils import *
from datasets import *
from models import *


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

yaml_path = "./config.yaml"
CONFIG = yaml_load(yaml_path)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    mode = command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    if mode:
        states = "train"
    else:
        states = "test"
    
    os.makedirs(CONFIG["model_directory"], exist_ok = True)
    csv_list = ["attributes_00.csv", "attributes_01.csv", "attributes_02.csv"]
    dir_list = select_dirs(config = CONFIG, mode = mode)
    for idx, target_dir in enumerate(dir_list):
        print("===============================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))
        
        root_path = os.path.split(target_dir)[0]
        machine_type = os.path.split(target_dir)[1]
        
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(
            model=CONFIG["model_directory"], machine_type=os.path.split(target_dir)[1]
        )
        
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=CONFIG["model_directory"],
                                                                                machine_type=machine_type)
        
        unique_section_names = numpy.unique(
            get_section_names(target_dir, dir_name="train")
        )
        
        n_sections = unique_section_names.shape[0]
        
        joblib.dump(unique_section_names, section_names_file_path)
        
        attr = []
        for c in csv_list:
            csv_path = os.path.join(root_path, machine_type, c)
            df = pd.read_csv(csv_path)
            
            for v in df["d1v"].unique():
                attr.append(v)
        
        n_classes = len(attr)
        
        print("============== DATASET_GENERATOR ==============")
        
        # n_files_ea_section = []
        # total_file_path =[]
        # data = np.empty((0, CONFIG["feature"]["n_frames"] * CONFIG["feature"]["n_mels"]), float)
        
        
        # for section_idx, section_name in enumerate(unique_section_names):
        #     files, _ = file_list_generator(
        #             target_dir=target_dir,
        #             section_name=section_name,
        #             dir_name="train",
        #             mode=mode,
        #         )
        
        #     for p in files:
        #         total_file_path.append(p)
                  
        #     n_files_ea_section.append(len(files))

        #     data_ea_section = file_list_to_data(files,
        #                                         msg="generate train_dataset",
        #                                         n_mels=CONFIG["feature"]["n_mels"],
        #                                         n_frames=CONFIG["feature"]["n_frames"],
        #                                         n_hop_frames=CONFIG["feature"]["n_hop_frames"],
        #                                         n_fft=CONFIG["feature"]["n_fft"],
        #                                         hop_length=CONFIG["feature"]["hop_length"],
        #                                         power=CONFIG["feature"]["power"])

        #     data = np.append(data, data_ea_section, axis=0)

        # number of all files
        # n_all_files = sum(n_files_ea_section)
        # number of vectors for each wave file
        # n_vectors_ea_file = int(data.shape[0] / n_all_files)
        #total_file_path
        
        # make one-hot vector for conditioning
        # condition = np.zeros((n_sections, n_all_files), float)
        
        # start_idx = 0
        # for section_idx in range(n_sections):
        #     n_vectors = n_vectors_ea_file * n_files_ea_section[section_idx]
        #     condition[start_idx : start_idx + n_vectors, section_idx : section_idx + 1] = 1
        #     start_idx += n_vectors
            
        
        clf_dataset = WavMelClassifierDataset(root_path, 
                                              machine_type, 
                                              csv_list, 
                                              CONFIG["feature"]["sr"],
                                              states)
        
        train_clf_dataset = clf_dataset.get_dataset(n_mels=CONFIG["feature"]["n_mels"],
                                                    n_fft=CONFIG["feature"]["n_fft"],
                                                    hop_length=CONFIG["feature"]["hop_length"],
                                                    win_length=CONFIG["feature"]["n_fft"],
                                                    power=CONFIG["feature"]["power"])
        
        
        print("\n=========== DATALOADER_GENERATOR ==============")
        data_loader = {"train": None, "val": None, "eval_train": None}
        (
            data_loader["train"],
            data_loader["val"],
            data_loader["eval_train"],
        ) = get_dataloader(train_clf_dataset, CONFIG)
        print("===============================================")
        
        
        #define model
        arcface = ArcMarginProduct(128, n_classes, m=CONFIG["arcface"]["m"], s=CONFIG["arcface"]["s"])
        model = STgramMFN(num_class=n_classes,
                      c_dim=CONFIG["feature"]["n_mels"],
                      win_len=CONFIG["feature"]["n_fft"],
                      hop_len=CONFIG["feature"]["hop_length"],
                      arcface=arcface).to(DEVICE)
        
        optimizer = optim.Adam(
            params=model.parameters(),
            weight_decay=CONFIG["fit"]["weight_decay"],
            lr=CONFIG["fit"]["lr"],
        )   

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=CONFIG["fit"]["lr_step_size"],
            gamma=CONFIG["fit"]["lr_gamma"],
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(1, CONFIG["fit"]["epochs"]+1):
            now = datetime.datetime.now()
            now_str = now.strftime("%Y/%m/%d %H:%M:%S")
            print("{} Epoch {:2d} Train: ".format(now_str, epoch), end="")
            train_acc, train_loss = training(model, data_loader = data_loader["train"], criterion = criterion, optimizer = optimizer, DEVICE=DEVICE, epoch = epoch, scheduler=scheduler)
            now = datetime.datetime.now()
            now_str = now.strftime("%Y/%m/%d %H:%M:%S")
            print("{} Epoch {:2d} Valid: ".format(now_str, epoch), end="")
            validation(model, data_loader["val"], criterion = criterion, DEVICE = DEVICE, CONFIG = CONFIG)
            
        del train_clf_dataset, data_loader
        
if __name__ == "__main__":
    main()