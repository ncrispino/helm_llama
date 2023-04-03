from get_datasets import helm_dataset_map
from eval import brute_force_eval

def evaluate_dataset(data_id):   
    labels = {1: True, 2: True, 3: False, 4: True, 5: False, 6: False, 7: True, 8: False, 9: False, 10: False, 11: True, 12: False, 13: False, 14: False, 15: True, 16: False, 17: False, 18: False, 19: False, 20: False, 21: False, 22: False, 23: False, 24: False, 25: False, 26: True, 27: False, 28: False, 29: False, 30: False, 31: False, 32: False, 33: False, 34: False, 35: False, 36: False, 37: False, 38: True, 39: False, 40: True, 41: False, 42: True, 43: False, 44: False, 45: False, 46: True, 47: True, 48: False, 49: False, 50: False, 51: False, 52: False, 53: False, 54: False, 55: False, 56: False, 57: False, 58: False, 59: True, 60: False, 61: False, 62: True, 63: True, 64: False, 65: False, 66: False, 67: True, 68: False, 69: True, 70: False, 71: True, 72: False, 73: False, 74: True, 75: True, 76: False, 77: True, 78: False, 79: False, 80: False, 81: False, 82: False, 83: False, 84: True, 85: False, 86: False, 87: False, 88: False, 89: False, 90: False, 91: False, 92: False, 93: False, 94: False, 95: False, 96: False, 97: True, 98: False, 99: False, 100: False} 
    data_name = helm_dataset_map(data_id)    
    dataset_file = f"/scratch/cnicholas/helm/dataset/{data_name}"
    for i in [1, 4, 6, 7, 9]:
        for k in [0, 5, -5]:
            score = brute_force_eval(labels, dataset_file)
            print(f"Running {data_id} with prompt {i} and {k} tokens prepended.\n\tScore: {score:.2f}")