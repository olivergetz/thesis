import torch
from dataset import DatasetUCS
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score
import os
import numpy as np
import pandas as pd
import json

def k_fold_zs_evaluation(true, pred):
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    average_precision = average_precision_score(np.array(true).reshape(-1, 1), np.array(pred).reshape(-1, 1))
    
    return accuracy, precision, recall, f1, average_precision

def k_fold_zs_predict(settings, audio_embeddings:torch.Tensor, text_encoding_func, similarity_func, device, topk:int=1, merge_classes=list[list]):
    """
    This function evaluates pytorch audio embeddings on zero-shot classification.

    Args:
    - settings: object containing configurations for the experiment.
    - audio_embeddings: extracted pytorch embeddings, shape (batch_size, n_classes, latent_dim). Indeces must correspond to the ground truth csv.
    - ground_truth_path: path to directory containing .csv files, one for each fold
    - text_encoding_func: the function to use to extract text embeddings
    - similarity_func: function to be used to compute similarity. Must take two sets of embeddings as input.
    - k_folds: must be equal or less than the number of folds
    - topk: output the top-k predictions. Adds an extra dimension to the prediction output.
    """
    from tqdm import tqdm

    # Load USC Classes
    with open (settings['class_converter'], 'r') as f:
        ucs_classes = json.load (f)

    ground_truth_path = settings['ground_truth']

    fold_csv_list = [file for file in os.listdir(ground_truth_path) if file[-4:] == ".csv"]

    pred = []
    true = []
    for (true_fold, pred_fold) in tqdm(zip(fold_csv_list, audio_embeddings), total=len(fold_csv_list)):
        # Load fold from csv, used to compare class 
        df = pd.read_csv(ground_truth_path + true_fold)
        dataset = DatasetUCS(df, settings, device=device, return_type='path')
        label_embeddings = text_encoding_func(list(ucs_classes['class_to_int'].keys()))

        for i in tqdm(range(len(dataset)), leave=False):
            _, _, class_name = dataset[i] # gold label
            
            current_emb = pred_fold[i].to(device)
            similarities = similarity_func(current_emb, torch.Tensor(label_embeddings).to(device))
            similarities = similarities.detach()
            current_emb_probs = torch.softmax(similarities.unsqueeze(0), dim=1).detach().flatten() * 100

            if (topk > 1):
                idx = torch.topk(current_emb_probs, k=topk).indices
                pred.append(idx.tolist())
            else:
                idx = torch.argmax(current_emb_probs)
                pred.append(idx.item())

            true.append(ucs_classes['class_to_int'][class_name])

    return true, pred

def load_embeddings_from_disk(dir_path:str):
    """
    Loads a list of torch embeddings from a supplied directory and concatenates them along the first dimension. Output shape: (n_embeddings, (tensor.shape))

    Args:
    - dir_path: path to a directory containing pytorch embeddings. These embeddings must be of the same shape.
    """
    audio_embeddings = None
    dir = [file for file in os.listdir(dir_path) if file[-3:] == ".pt"]
    for file in dir:
        if (audio_embeddings != None):
            audio_embeddings = torch.cat((audio_embeddings, torch.load(dir_path + file).unsqueeze(0)), dim=0)
        else:
            audio_embeddings = torch.load(dir_path + file).unsqueeze(0)

    return audio_embeddings