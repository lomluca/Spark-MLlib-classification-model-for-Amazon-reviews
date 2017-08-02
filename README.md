# Spark MLlib classification model for Amazon reviews 
In this script I try two classification model (Logistic Regression and Decision Tree) to label Amazon reviews as "useful" or "useless". The precision of the Decision Tree is about 90% :thumbsup:

From the Amazon fine-foods dataset I extract the following features:
- Lenght of the review text
- Number of reviews of the user
- Number of useful reviews of the user
- Score of the review

## Test results
Output of the script:
```
Confusion matrix LR:
16067.0  16775.0
5521.0   51505.0  

Confusion matrix DC: 
54458.0  2568.0
5729.0   27113.0  

Precision LR = 0.7519027907597811                                               
Precision DC = 0.9076757021409178
```

## Dataset
[Amazon fine-foods dataset](https://snap.stanford.edu/data/web-FineFoods.html)
