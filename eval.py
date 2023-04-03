import json
import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.metrics.scores import f_measure

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def f1_score(gold: str, pred: str) -> float:
    ret = f_measure(set(normalize_text(gold).split()), set(normalize_text(pred).split()))
    if ret is None:  # answer is the empty string after normalizing
        return 0.0

    return ret

def exact_match(gold: str, pred: str) -> float:
    if not pred:
        return 0

    return 1 if gold.strip() == pred.strip() else 0

def quasi_prefix_exact_match(gold: str, pred: str) -> float:
    """
    `prefix_exact_match` after normalize the text. 
    """
    if not pred:
        return 0

    return 1 if normalize_text(pred).startswith(normalize_text(gold)) else 0

def readJson(fileName: str):
    with open(fileName, encoding='utf-8') as f:
        text = f.read()

    rex1 = "\"f1_score\": (.*?)\n"
    scores = re.findall(rex1, text, re.S)
    scores = np.array(scores).astype(float)

    print("f1:", np.mean(scores))

def compareJson(goal_filename: str, pred_filename: str, metric: str) -> float:
    """
        compareJson(goal_filename, pred_filename, metric)
        metric: "f1" / "em" / "qem"
    """
    with open(goal_filename,'r') as f1:
        goal_f = json.loads(f1)
    with open(pred_filename, 'r') as f2:
        pred_f = json.loads(f2)    

    if len(list(goal_f.keys())) != len(list(pred_f.keys())):
        return -1

    scores = []
    for i in range(len(list(goal_f.keys()))):
        golds = goal_f[i]['answers']
        if 'output_mapping' in list(goal_f[i].keys()):
            map = goal_f[i]['output_mapping']
        pred = pred_f[i].split("\nAnswer:")[-1]
        # truncate after \n\n
        truncate = pred.find("\n\n")
        if truncate != -1:
            pred = pred[0: truncate]
        

        score = -1
        for gold in golds:
            if metric == "f1":
                score = max(score, f1_score(gold, pred))
            elif metric == "em":
                score = max(score, exact_match(gold, pred))
            elif metric == "qem":
                score = max(score, quasi_prefix_exact_match((gold, pred)))
            else:
                return -1           
        if score == -1:
            return -1
        scores.append(score)
    
    return np.mean(np.array(scores).astype(float))

def brute_force_eval(labels, dataset_file):
    # labels = get_helm_labels(f"/scratch/cnicholas/helm/dataset/labels_{dataset_file}", num_instances = num_instances)
    #with open(f"labels_{dataset_file}", "w") as labels_json:
    #    json.dump(labels, labels_json)
    # Calculate naive accuracy.
    with open(dataset_file, 'r') as f2:
        pred_f = json.loads(f2)
    print("labels: ", labels)
    print("pred f: ", pred_f)
    num_instances = len(pred_f)
    num_correct = 0
    for i in range(num_instances):
        if pred_f[i] == labels[i]:
            num_correct += 1
    return num_correct/num_instances

    

def eval():
    prompts = [1, 2, 3, 4, 5]

    data = {}
    for pid in prompts:
        p = []
        p.append(compareJson("boolq.json", "2_" + pid + ".json", "qem"))
        p.append(compareJson("truthful_qa", "3_" + pid + ".json", "em"))
        p.append(compareJson("natural_qa_closed", "4_" + pid + ".json", "f1"))
        p.append(compareJson("natural_qa_open", "5_" + pid + ".json", "f1"))
        p.append(compareJson("quac", "6_" + pid + ".json", "f1"))
        p.append(compareJson("commonsense_hellaswag", "7_" + pid + ".json", "em"))
        p.append(compareJson("commonsense_openbookqa", "8_" + pid + ".json", "em"))
        p.append(compareJson("narrative_qa", "9_" + pid + ".json", "f1"))

        data[str(pid)] = p
    
    df = pd.DataFrame(data)
    df.to_csv("results.csv")

if __name__ == "__main__":
    print(f1_score("hello hi", "hello ha"))
    # print(exact_match("hello hi", "hello ha"))
    # print(quasi_prefix_exact_match("hello", "hello ha"))
