# ML (DT model and evaluation)
FOLDS = 5

# Rule learning 
# obj = lamb1 * precision  +  lamb2 * recall  +  lamb3 * (1/size(rules))e

alpha = 0.05 # Mann-Whitney score threshold
delta = 60 # Threshold for rule learning coverage
lamb1 = 1 # scaling factor on precision
lamb2 = 1 # scaling factor on recall
lamb3 = 1 # scaling factor on rule size

