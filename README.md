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

# Problem Statement:

The task involves using machine learning to forecast future values in a time series dataset. This includes defining evaluation metrics, splitting data for training, validation, and testing, and tuning hyperparameters systematically for each hypothesis class. The main goal is to identify the best hypothesis class for the prediction task. We evaluate various hypothesis classes such as autoregressive models, LSTM networks, gradient boosting (e.g., HG Boost), linear regression, and decision trees (e.g., Random Forests), each with unique strengths and weaknesses affecting their predictive abilities in time series analysis. The assessment involves recognizing common errors in each model type, such as linear models struggling with nonlinear relationships and decision trees facing overfitting if not pruned properly. Neural networks, while adept at capturing temporal dependencies, may encounter computational complexity and training instability.

Proposed solutions include feature engineering to capture nonlinear relationships, regularization to prevent overfitting, and designing architectures tailored to the data's characteristics. Through systematic evaluation and refinement, we aim to determine the hypothesis class that best captures the patterns and dependencies in the time series data, improving predictions' accuracy and reliability.

# Methodology :

In our study, we aim to identify the best hypothesis class for predicting Delhi's temperature using historical data from 1995 to 2020. Our goal is to develop reliable models that can forecast future temperature accurately, crucial for applications like climate research and urban planning. We start by framing the problem as a regression task, where we estimate future temperatures based on past observations. Evaluation metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) help assess model performance.

Data preprocessing involves extracting features from historical temperature data and splitting it into training, validation, and test sets based on a time-based strategy. We then evaluate three primary hypothesis classes: linear models, decision trees, and neural networks with a focus on Long Short-Term Memory (LSTM). Hyperparameter optimization is conducted through techniques like grid search or random search to fine-tune model parameters. After optimization, we assess each model's performance on the validation set to identify the most promising class. Error analysis highlights strengths and weaknesses, with proposed solutions including feature engineering, regularization, pruning, and architecture optimization.

Finally, we rigorously evaluate models on the test set to determine the best-performing hypothesis class. Our study concludes with synthesized findings and recommendations for future research, aiming to advance temperature prediction methodologies across various domains.

# Experimental Results and Validaton :

The dataset consists of daily temperature records from Delhi spanning 1995 to 2020, sourced from the India Meteorological Department. Before analysis, missing values were handled by interpolation or carrying forward the last observed value. For feature engineering, a sliding window approach was employed to convert the time series data into a supervised learning problem, utilizing lagged features where the previous 24-hour temperatures predict the next day's temperature. Model selection involved various hypothesis classes: Linear Regression, XGBoost, ARIMA, Random Forest, and LSTM, each addressing the forecasting problem with distinct strengths. Performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) were used for evaluation.

# Conclusions and Future Work :-

In the domain of temperature forecasting for Delhi, LSTM emerges as the standout performer. Its remarkable ability to capture intricate patterns within the data sets it apart from other models. While ARIMA demonstrates competitive performance, LSTM's proficiency in handling long-term relationships gives it a distinct advantage. XGBoost initially shows promise with impressive results, but its struggle to generalise poses a limitation. On the other hand, both linear regression and random forest models must catch up in capturing the nuanced complexities of the temperature data. In summary, while other models exhibit potential, LSTM's exceptional capacity to learn from sequential data makes it the premier choice for accurate temperature predictions in Delhi. Its robust performance and superior adaptability solidify its position as the preferred solution for achieving the region's precise and reliable temperature forecasts.

Moving ahead, various avenues exist to advance temperature forecasting in Delhi. Exploring hybrid models, like combining LSTM with ARIMA, could improve predictions. Integrating external factors such as weather patterns and urbanisation trends may enhance model performance. Investigating advanced deep learning architectures beyond LSTM, such as attention-based models, could better capture long-range dependencies. Additionally, ensemble techniques can mitigate model biases and improve forecasting accuracy. Evaluating model robustness under diverse climatic conditions is essential for real-world applications. These efforts aim to enhance temperature forecasting accuracy, contribute to climate resilience, and aid in urban planning strategies for Delhi.

