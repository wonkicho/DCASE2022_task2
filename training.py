import datetime
import enum
import os
import sys
from itertools import chain

import wandb
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


yaml_path = "./config.yaml"
CONFIG = yaml_load(yaml_path)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', DEVICE)  # 출력결과: cuda 
print('Count of using GPUs:', torch.cuda.device_count())   #출력결과: 1 (GPU #2 한개 사용하므로)
print('Current cuda device:', torch.cuda.current_device())

def main():
    mode = command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    if mode:
        states = "train"
    else:
        states = "test"
    
    os.makedirs(CONFIG["model_directory"], exist_ok = True)
    csv_list = ["attributes_00.csv", "attributes_01.csv", "attributes_02.csv", "attributes_03.csv", "attributes_04.csv", "attributes_05.csv"]
    dir_list = select_dirs(config = CONFIG, mode = mode)
    for idx, target_dir in enumerate(dir_list):
        print("===============================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))
        
        root_path = os.path.split(target_dir)[0]
        machine_type = os.path.split(target_dir)[1]
        
        run = wandb.init(project = "DCASE-Anomaly_Detection_MFN_condition4", name = f"{machine_type}", config = CONFIG, entity = "chowk")

        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(
            model=CONFIG["model_directory"], machine_type=os.path.split(target_dir)[1]
        )
        
        score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=CONFIG["model_directory"],
                                                                                machine_type=machine_type)
        
        unique_section_names = np.unique(
            get_section_names(target_dir, dir_name="train")
        )
        eval_section_names = np.unique(
            get_section_names(target_dir, dir_name="test")
        )
        n_sections = unique_section_names.shape[0]


        unique_section_list = unique_section_names.tolist()
        eval_section_list = eval_section_names.tolist()

        
        
        joblib.dump(unique_section_names, section_names_file_path)
        
        attr = []
        attr_dict = {}
        
        
        sec_dict = {}
        for c in csv_list:
            csv_path = os.path.join(root_path, machine_type, c)
            df = pd.read_csv(csv_path)
            
            for f, v in zip(df["file_name"], df["d1v"]):
                if len(f.split('/')) == 1 and (machine_type == "ToyCar" or machine_type == "ToyTrain"):
                    f = os.path.join(machine_type, "train", f)

                sec_dict[f] = re.findall('section_[0-9][0-9]', f)[0]
                attr_dict[f] = v

            for v in df["d1v"].unique():
                attr.append(v)
        
        n_classes = len(unique_section_list)#len(attr)
        print(f'class num : {n_classes}')

        

        tr_clf_dataset = WavMelClassifierDataset(root_path, 
                                              machine_type, 
                                              csv_list, unique_section_list,
                                              CONFIG["feature"]["sr"],
                                              states)
        
        train_clf_dataset = tr_clf_dataset.get_dataset(n_mels=CONFIG["feature"]["n_mels"],
                                                    n_fft=CONFIG["feature"]["n_fft"],
                                                    hop_length=CONFIG["feature"]["hop_length"],
                                                    win_length=CONFIG["feature"]["win_length"],
                                                    power=CONFIG["feature"]["power"])
        
        
        train_loader = torch.utils.data.DataLoader(
                                                train_clf_dataset,
                                                batch_size=CONFIG["fit"]["batch_size"],
                                                shuffle=CONFIG["fit"]["shuffle"],
                                                drop_last=True,
                                            )


        print("===============================================")
        
        
        #define model
        arcface = ArcMarginProduct(128, n_classes, m=CONFIG["arcface"]["m"], s=CONFIG["arcface"]["s"])
        model = STgramMFN(num_class=n_classes,
                      c_dim=CONFIG["feature"]["n_mels"],
                      win_len=CONFIG["feature"]["win_length"],
                      hop_len=CONFIG["feature"]["hop_length"],
                      arcface=arcface).to(DEVICE)
        
        optimizer = optim.Adam(
            params=model.parameters(),
            weight_decay=CONFIG["fit"]["weight_decay"],
            lr=CONFIG["fit"]["lr"],
        )   

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG["fit"]["epochs"]/10, eta_min=CONFIG["fit"]["lr"] / 10,
                                                           last_epoch=-1)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        wandb.watch(model, criterion, log = "all", log_freq = 10)
        best_auc = 0
        for epoch in range(1, CONFIG["fit"]["epochs"]+1):
            now = datetime.datetime.now()
            now_str = now.strftime("%Y/%m/%d %H:%M:%S")
            print("{} Epoch {:2d} Train: ".format(now_str, epoch), end="")
            train_loss = training(model, data_loader = train_loader, criterion = criterion, optimizer = optimizer, DEVICE=DEVICE, epoch = epoch, config = CONFIG, scheduler=scheduler)
            now = datetime.datetime.now()
            now_str = now.strftime("%Y/%m/%d %H:%M:%S")
            print("{} Epoch {:2d} Valid: ".format(now_str, epoch), end="")
            val_loss , mean_auc, mean_p_auc = evaluation(model = model, target_dir = target_dir, unique_section_names= eval_section_names,
                                            machine_type=machine_type, sec = unique_section_list, sec_dict = sec_dict,
                                            criterion = criterion, CONFIG=CONFIG, DEVICE=DEVICE, mode=mode)
            
            print(f'{machine_type}\t[{epoch}/{CONFIG["fit"]["epochs"]}]\tValidation Loss: {val_loss:3.3f}\t[AUC]: {mean_auc:3.3f}\t[pAUC]: {mean_p_auc:3.3f}')
            
            model_checkpoints = {
                "epoch" : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
            
            if (mean_auc + mean_p_auc) > best_auc:
                best_auc = mean_auc + mean_p_auc
                print("\nBest AUC SCORE CHANGED")
                print(f"BEST EPOCH : {epoch} || BEST AUC : {mean_auc} || BEST pAUC : {mean_p_auc}")
                save_model(model, model_dir=CONFIG["model_directory"], machine_type = machine_type, states = model_checkpoints)
            
            wandb.log({"Epoch" : epoch, "Train Loss" : train_loss, "Valid Loss" : val_loss, "AUC" : mean_auc, "pAUC" : mean_p_auc})

        del train_clf_dataset, train_loader

        fit_gamma_dist(model = model, target_dir = target_dir, machine_type = machine_type, sec = unique_section_list, sec_dict = sec_dict, CONFIG=CONFIG, DEVICE=DEVICE,mode=mode)
        run.finish()
        print("============== END TRAINING ==============")

if __name__ == "__main__":
    main()