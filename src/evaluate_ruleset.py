import pandas as pd

def evaluate_ruleset(df, ruleset)
    #TODO replace, from chatGPT
    combined_prediction = np.zeros(len(data))
        for rule in ruleset:
            combined_prediction |= np.array([rule(sample) for sample in data])

        true_positive = np.sum((combined_prediction == 1) & (data[:, -1] == 1))
        false_positive = np.sum((combined_prediction == 1) & (data[:, -1] == 0))
        false_negative = np.sum((combined_prediction == 0) & (data[:, -1] == 1))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        return precision, recall
