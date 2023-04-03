import argparse

def compare_json(x, y):
    """Given two json files structured like {"1": ["True"], "2": ["False"]}, returns the number of correct predictions."""
    with open(x) as f:
        x = json.load(f)
    with open(y) as f:
        y = json.load(f)
    correct = 0
    for key in x:
        if x[key] == y[key]:
            correct += 1
    return correct/len(x)

if __name__=="__main__":
    argparse
    
