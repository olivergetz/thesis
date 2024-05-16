import torch
from utils.dataset import DatasetUCS
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, f1_score
import os
import numpy as np
import pandas as pd
import json
from typing import Literal

def precision_at_k(x):
    """
    Iterates over a list sorted from most to least relevant and returns the cumulative precision at each step.

    Args:
        x: one-hot list of relevant hits. Must be ordered by relevance.
    """
    n_hits = 0
    prec_k = []
    for i, relevance in enumerate(x):
        if relevance == 1:
            n_hits += 1

        prec_k.append(n_hits / (i+1))

    return n_hits, prec_k

def mean_avg_precision_at_k(query_results):
    """
    Iterates over a list of query results, one-hot encoded by relevance.

    Args:
        query_results: list of one-hot encodings marking the relevance of a query result. Sub-lists must be ordered by relevance.
    """
    import math
    ap_all_queries = []
    for result in query_results:
        # precision@k
        n_hits, prec_k = precision_at_k(result)
        hit_prec_k = np.array(result) * np.array(prec_k)
        ap = hit_prec_k.sum() / n_hits
        if (math.isnan(ap)):
            ap_all_queries.append(0)
        else:
            ap_all_queries.append(ap.sum() / n_hits)

    return np.array(ap_all_queries).mean()

def semantic_search(prompt:str, n_return:int=10, audio_embeddings=None, model:Literal["msclap", "laion"]='msclap', device=torch.device):
    """
    Given a prompt, return audio files with the most similar semantic similarity. Based on LAION CLAP and MSCLAP.

    To do: shorten search time by limiting searches to the most likely category. Currently: 6.2 sec.

    Args.
    - prompt: A string containing the entire search query.
    - n_return: How many files the query should return.
    - audio_embeddings: The audio embeddings to compare with.

    Returns a list of indeces, ranked from most to least relevant.
    """
    from msclap import CLAP
    import laion_clap
    
    if (audio_embeddings == None): 
        print("Please provide a list of audio embeddings.")
        return
    
    # Extract text embeddings
    if (model == 'laion'):
        
        laion_clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        laion_clap_model.load_ckpt()
        laion_clap_model.to(device)
        text_data = ["", prompt] # CLAP has a bug where two text prompts are required. Discard the first.
        text_embedding = laion_clap_model.get_text_embedding(text_data)[1]
        cos = torch.cosine_similarity
    elif (model == 'msclap'):
        msclap_model = CLAP(version = '2023', use_cuda=True) # version can be 2022 or 2023
        text_embedding = msclap_model.get_text_embeddings([prompt])
        cos = msclap_model.compute_similarity
    else:
        print("The model must be either 'msclap' or 'laion'.")
        return

    audio_embeddings.to('cuda')
    similarities = torch.tensor([cos(embedding.clone().detach(), torch.tensor(text_embedding).clone().detach().to('cuda')) for embedding in audio_embeddings])
    idx = torch.argsort(similarities, dim=0, descending=True)

    return idx[:n_return]


def k_fold_zs_evaluation(true, pred):
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    average_precision = average_precision_score(np.array(true).reshape(-1, 1), np.array(pred).reshape(-1, 1))
    
    return accuracy, precision, recall, f1, average_precision

def k_fold_zs_predict(settings, audio_embeddings:torch.Tensor, text_encoding_func, similarity_func, device, topk:int=1, temp_aug:str=""):
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
    - temp_aug: augmentation to prepend to class names
    """
    from tqdm import tqdm

    # Load USC Classes
    with open (settings['class_converter'], 'r') as f:
        ucs_classes = json.load (f)

    # Augmentations
    if (temp_aug != ""):
        ucs_classes['class_to_int'] = {temp_aug+k:v for (k,v) in ucs_classes['class_to_int'].items()}
        ucs_classes['int_to_class'] = {k:temp_aug+v for (k,v) in ucs_classes['int_to_class'].items()}

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

            # Augment
            class_name = temp_aug + class_name
            
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


if __name__ == "__main__":

    relevant = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1], # Relevance for the top 10 query results for query 1. Getting a lot of weather, but not necessarily wind.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # etc ... 
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1], # Metallic. Results in a lot of musical sounds, glock, vibraphone, etc.
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # Lots of sizzling sounds, close to electricity.
        [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    ]

    print(mean_avg_precision_at_k(relevant))


    """
    from dataset import DatasetUCS
        # Load USC Classes
    with open ('C:/Users/olive/Documents/Programming_Environment/python_scripting/thesis/experiments/data/ucs_official/ucs_classes.json', 'r') as f:
        ucs_classes = json.load (f)

    temp_aug = "this is the sound of "

    # Augmentations
    class_to_int = ucs_classes['class_to_int']
    int_to_class = ucs_classes['int_to_class']

    print(ucs_classes['int_to_class'])
    ucs_classes['class_to_int'] = {temp_aug+k:v for (k,v) in ucs_classes['class_to_int'].items()}
    ucs_classes['int_to_class'] = {k:temp_aug+v for (k,v) in ucs_classes['int_to_class'].items()}
    print(ucs_classes['int_to_class'])
    """