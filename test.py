# Standard library imports.
import os
import gc
import sys
import csv
import joblib

# Related third party imports.
import numpy as np
import scipy.stats
import torch
import torch.utils.data
from scipy.special import softmax
from sklearn import metrics

from utils import *
from models import *
from datasets import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calc_decision_threshold(target_dir):
    """
    Calculate decision_threshold from anomaly score distribution.
    """

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=CONFIG["model_directory"], machine_type=os.path.split(target_dir)[1]
    )
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)
    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(
        q=CONFIG["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat
    )

    return decision_threshold

# Load configuration from YAML file.
CONFIG = yaml_load("./config.yaml")

# String constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mode = command_line_chk()  # constant: True or False
if mode is None:
    sys.exit(-1)


os.makedirs(CONFIG["result_directory"], exist_ok=True)

dir_list = select_dirs(config=CONFIG, mode=mode)

csv_lines = []
if mode:
    performance_over_all = []

performance = {"section": None, "all": None}
score_list = {"anomaly": None, "decision": None}

if mode == True:
    csv_list = ["attributes_00.csv", "attributes_01.csv", "attributes_02.csv","attributes_03.csv","attributes_04.csv","attributes_05.csv"]

if mode == False:
    csv_list = ["attributes_03.csv","attributes_04.csv","attributes_05.csv"]

for idx, target_dir in enumerate(dir_list):
    print("===============================================")
    print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

    print("================ MODEL LOAD =================")
    root_path = os.path.split(target_dir)[0]
    machine_type = os.path.split(target_dir)[1]
    section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(
        model=CONFIG["model_directory"], machine_type=machine_type
    )
    trained_section_names = joblib.load(section_names_file_path)
    n_sections = trained_section_names.shape[0]

    
    unique_section_list = trained_section_names.tolist()

    if mode:
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
        
    n_classes = n_sections
    


    arcface = ArcMarginProduct(128, n_classes, m=CONFIG["arcface"]["m"], s=CONFIG["arcface"]["s"])
    model = STgramMFN(num_class=n_classes,
                    c_dim=CONFIG["feature"]["n_mels"],
                    win_len=CONFIG["feature"]["win_length"],
                    hop_len=CONFIG["feature"]["hop_length"],
                    arcface=arcface).to(DEVICE)

    model_file = "{model}/model_{machine_type}.pth".format(
        model=CONFIG["model_directory"], machine_type=machine_type
    )
    model.eval()
    model.load_state_dict(torch.load(model_file))
    
    decision_threshold = calc_decision_threshold(target_dir)
    if mode:
        # results for each machine type
        csv_lines.append([machine_type])
        csv_lines.append(["", "AUC (source)", "AUC (target)", "pAUC", 
                        "precision (source)", "precision (target)", "recall (source)", "recall (target)",
                        "F1 score (source)", "F1 score (target)"])
        performance = []
    
    dir_name = "test"
    for section_name in get_section_names(target_dir, dir_name=dir_name):
        # search for section_name
        temp_array = np.nonzero(trained_section_names == section_name)[0]
        if temp_array.shape[0] == 0:
            section_idx = -1
        else:
            section_idx = temp_array[0]

        # load test file
        test_files, y_true = file_list_generator(
            target_dir=target_dir,
            section_name=section_name,
            dir_name=dir_name,
            mode=mode,
        )

        
            
        if mode:
            domain_list = []


        anomaly_score_list = []
        anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(result=CONFIG["result_directory"],
                                                                                                                machine_type=machine_type,
                                                                                                                section_name=section_name,
                                                                                                                dir_name=dir_name)
        decision_result_list = []
        decision_result_csv = "{result}/decision_result_{machine_type}_{section_name}_{dir_name}.csv".format(result=CONFIG["result_directory"],
                                                                                                                    machine_type=machine_type,
                                                                                                                    section_name=section_name,
                                                                                                                    dir_name=dir_name)
        
        
        print(
                "============== BEGIN TEST FOR A SECTION %s OF %s =============="
                % (section_name, dir_name))                                                                                                            


        y_pred = [0. for _ in test_files]
        for file_idx, file_path in enumerate(test_files):
            fp = os.path.split(file_path)[1]
            fp = os.path.join(machine_type, "test", fp)
            if mode:
                #label = attr.index(attr_dict[fp])
                #label = unique_section_list.index(sec_dict[fp])
                label = []
                for i in range(len(unique_section_list)):
                    if i == unique_section_list.index(sec_dict[fp]):
                        label.append(1)
                    else:
                        label.append(0)
                label = np.array(label)
                

            else:
                sec = re.findall('section_[0-9][0-9]', fp)[0]
                #label = unique_section_list.index(sec)
                label = []
                for i in range(len(unique_section_list)):
                    if i == unique_section_list.index(sec):
                        label.append(1)
                    else:
                        label.append(0)
                label = np.array(label)

                #label = label
            x_wav, x_mel, label = test_transform(file_path, label, CONFIG, DEVICE)
            with torch.no_grad():
                model.eval()
                predict_ids, feature = model(x_wav, x_mel, label)
                
            probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
            label = label.cpu()
            y_pred[file_idx] = probs[torch.argmax(label)]
            anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            
            # store decision results
            if y_pred[file_idx] > decision_threshold:
                decision_result_list.append([os.path.basename(file_path), 1])
            else:
                decision_result_list.append([os.path.basename(file_path), 0])

            if mode:
                domain_list.append("source" if "source" in file_path else "target")

        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

        save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
        logger.info("decision result ->  {}".format(decision_result_csv))

        if mode:
            # extract scores used for calculation of AUC (source) and AUC (target)
            y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="source" or y_true[idx]==1]
            y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="source" or y_true[idx]==1]
            y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx]=="target" or y_true[idx]==1]
            y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx]=="target" or y_true[idx]==1]

            # calculate AUC, pAUC, precision, recall, F1 score 
            auc_s = metrics.roc_auc_score(y_true_s, y_pred_s)
            auc_t = metrics.roc_auc_score(y_true_t, y_pred_t)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=CONFIG["max_fpr"])
            tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]).ravel()
            tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, [1 if x > decision_threshold else 0 for x in y_pred_t]).ravel()
            prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
            prec_t = tp_t / np.maximum(tp_t + fp_t, sys.float_info.epsilon)
            recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
            recall_t = tp_t / np.maximum(tp_t + fn_t, sys.float_info.epsilon)
            f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)
            f1_t = 2.0 * prec_t * recall_t / np.maximum(prec_t + recall_t, sys.float_info.epsilon)

            csv_lines.append([section_name.split("_", 1)[1],
                            auc_s, auc_t, p_auc, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

            performance.append([auc_s, auc_t, p_auc, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
            performance_over_all.append([auc_s, auc_t, p_auc, prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])

            logger.info("AUC (source) : {}".format(auc_s))
            logger.info("AUC (target) : {}".format(auc_t))
            logger.info("pAUC : {}".format(p_auc))
            logger.info("precision (source) : {}".format(prec_s))
            logger.info("precision (target) : {}".format(prec_t))
            logger.info("recall (source) : {}".format(recall_s))
            logger.info("recall (target) : {}".format(recall_t))
            logger.info("F1 score (source) : {}".format(f1_s))
            logger.info("F1 score (target) : {}".format(f1_t))

        print("\n============ END OF TEST FOR A SECTION ============")
    
    if mode:
        # calculate averages for AUCs and pAUCs
        amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
        csv_lines.append(["arithmetic mean"] + list(amean_performance))
        hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
        csv_lines.append(["harmonic mean"] + list(hmean_performance))
        csv_lines.append([])

    del model
    gc.collect()

if mode:
    csv_lines.append(["", "AUC (source)", "AUC (target)", "pAUC", 
                    "precision (source)", "precision (target)", "recall (source)", "recall (target)",
                    "F1 score (source)", "F1 score (target)"])     

    # calculate averages for AUCs and pAUCs
    amean_performance = np.mean(np.array(performance_over_all, dtype=float), axis=0)
    csv_lines.append(["arithmetic mean over all machine types, sections, and domains"] + list(amean_performance))
    hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance_over_all, dtype=float), sys.float_info.epsilon), axis=0)
    csv_lines.append(["harmonic mean over all machine types, sections, and domains"] + list(hmean_performance))
    csv_lines.append([])
    
    # output results
    result_path = "{result}/{file_name}".format(result=CONFIG["result_directory"], file_name=CONFIG["result_file"])
    logger.info("results -> {}".format(result_path))
    save_csv(save_file_path=result_path, save_data=csv_lines)