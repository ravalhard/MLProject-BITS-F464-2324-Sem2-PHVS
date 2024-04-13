# MLProject-BITS-F464-2324-Sem2-PHVS

**BITS F464 Machine Learning Project**
**Instructors :** Aditya Challa and Ashwin Srinivisan

**Team Members :-**
1. Vaibhav Jain [2021A7PS2389G]
2. Hardav Raval [2021A7PS3041G]
3. Apurva Patil [2021A7PS2068G]
4. Saksham Tripathi [2021A7PS2477G]

**Choosen Project Statement :-**
Which hypothesis classes are better tuned for time series prediction?
A time series datasetis nothing but a data in the form x1, x2, · · · xT , where each of the xi ∈ Rd. 
This can be converted to a supervised regression problem by asking - Can you predict the value next time given the previous k
values?. That is we want to identify the relationship

  xt−k, xt−k+1, · · · , xt−1 → xt. (1)

So, the aim is to identify the right hypothesis class to do this problem?
1. First, setup the ML problem correctly - (i) What is the right metric to evaluate the model? (ii)
How should you do the train/valid/test splits? etc.
2. For each model of hypothesis class, do a hyper-parameter optimization and accordingly select the
best model.
3. Comment on the performance of linear models, decision trees and neural networks for time series.
Can you identify where the errors in these models are coming from? and propose a solution?
