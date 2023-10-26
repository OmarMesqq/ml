## ML Notes:
### Cross entropy:
A measure of the "difference" between two probability distributions. 
For a true distribution `p`, we have
**1** as the the true class probability and **0** for all other classes.
`q` is the predicted probability from the model.

The mathematical definition for it could be:

$$ H(p, q) = -\sum_{i=0}^{N-1} p(i) \ln(q(i)) $$

where 
- we are summing over a series of terms each corresponding to a different category/class `i`, 
from `i = 0` to `i = N -1`.
- `p(i)` and `q(i)` are the true and model-predicted probability of category `i`, respectively.

Cross entropy will be low if true and predicted probabilities are close. 
Otherwise, it will be higher. 

### One-hot encoded arrays
For each value, a binary vector of all zeros except for a one at the index corresponding to the value is created. Good for representing categorical data.
For example, in a problem where you have three classes (0, 1, and 2), the one-hot encoding would be:

```
Class 0: [1, 0, 0]
Class 1: [0, 1, 0]
Class 2: [0, 0, 1]
```

### Categorical data:
Variables that can take on one of a limited, and usually fixed, number of possible values.
Example: education level - categories could be high school, bachelor's degree, master's degree, etc.
