# loan_default_prediction_using_PCA

Hi everyone, I'm Anurag from the Indian Statistical Institute. Today, I'll be walking you through a series of code focusing on Principal Component Analysis (PCA) using a notebook from a Kaggle competition hosted by Imperial College London in 2014.

Video Tutorial:  
[<img width="660" alt="Screenshot 2024-05-08 at 3 42 27 PM" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/075d6d83-f523-4533-98c7-99a4c43f404c">](https://www.youtube.com/watch?v=gjHHIFlU2E8 "Play on YouTube")

<h3>Aim:</h3>
<div>The competition aimed to predict whether a loan would default and the potential loss if it did. Unlike traditional methods that categorize borrowers as good or bad, this competition sought to anticipate both the default likelihood and the severity of losses. Essentially, it aimed to bridge traditional banking methods with asset management strategies.</div>

<h3>Motivation:</h3>
<div>
  We chose the loan default dataset for two main reasons:
<ul><li>Real-World Impact: The issue of loan default has significant real-world implications for both lenders and borrowers. By analyzing this dataset, we can develop models and strategies to better predict and prevent loan defaults. This can help financial institutions minimize their losses and make more informed lending decisions, ultimately contributing to a more stable financial system.</li>
  
<li>Complexity and Challenge: The loan default prediction task presents a complex and challenging problem for data analysis and machine learning. The dataset likely contains a wide range of variables that influence loan default, such as borrower demographics, financial history, loan terms, economic factors, and more. Tackling this dataset requires sophisticated analytical techniques, including feature engineering, dimensionality reduction (like PCA), and predictive modeling (such as logistic regression or random forests). Successfully addressing these challenges can enhance our skills as data scientists and provide valuable insights into predictive modeling in finance.</li></ul>

In essence, the loan default dataset offers both practical relevance and intellectual challenge, making it a compelling choice for analysis and exploration.

</div>
<h3>Importing Libraries</h3>
<pre>import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt</pre>

<h3>Loading Dataset</h3>
<pre>
  train_data = pd.read_csv("/content/drive/MyDrive/Loan_Default/train_v2.csv")
</pre>
<samp><img width="1293" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/9edafd0a-075a-4b69-a9fd-ec3527a7ba01">
</samp>

<pre>
  test_data = pd.read_csv("/content/drive/MyDrive/Loan_Default/test_v2.csv")
</pre>
<samp>
  <img width="1205" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/e9cfbe9f-cfdf-4808-8338-5e8107e35c08">
</samp>
<h3>View DataFrame</h3>
<h4>Train Data</h4>
<pre>
  train_data.head()
</pre>
<samp>
  <img width="1143" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/3ae1c2d8-75b0-4f3a-8293-c9e8a6b475bf">
</samp>
<h3>Test Data</h3>
<pre>test_data.head()</pre>
<samp><img width="1183" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/205cd7e5-df9c-4e13-8446-bb66849c7775">
</samp>
<h3>Data Information</h3>
<pre>train_data.info()</pre>
<samp>
  <img width="422" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/bf6d9c7b-adc9-46f8-93a7-0e7d38a3bbc0">
</samp>
<pre>test_data.info()</pre>
<samp>
  <img width="507" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/1ff46dbb-341c-47c4-b906-0b769dc2e9c6">
</samp>
<h3>Missing Values/Null Values</h3>
<pre>sns.heatmap(train_data.isnull(), cbar=False)</pre>
<samp>
  <img width="631" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/b48a4f7a-3dac-40ee-91d2-455cae1dd623">
</samp>
<pre>
  missing = train_data.isnull().sum()
missing = pd.DataFrame(missing[missing!=0])
missing.columns = ['No. of missing values']
missing['Percentage'] = 100*missing['No. of missing values']/train_data.id.count()
missing.sort_values(by="Percentage", ascending=False)
</pre>
<samp>
  <img width="744" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/80f7bf3d-45e4-4f52-af60-2fcfb9713772">
</samp>
<h3>Drop columns having incorrect dtype</h3>
<pre>
  train_data.select_dtypes(include=['object']).head()
</pre>
<samp>
  <img width="921" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/43b20124-afb8-4d9c-b5d5-e1d03653c059">
</samp>
<pre>
invalid = train_data.select_dtypes(include=['object']).columns
train_data.drop(invalid, axis=1, inplace=True)
</pre>
<pre>
test_data.drop(invalid, axis=1, inplace=True)
t_id = test_data['id'].copy
test_data.drop('id', axis=1, inplace = True)
</pre>
<h3>Describe Statistics</h3>
<pre>train_data.describe()</pre>
<samp>
  <img width="891" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/b45fb32d-2cd2-464c-a6c7-e3c43f6a2814">
</samp>
<pre>test_data.describe()</pre>
<samp>
  <img width="891" alt="image" src="https://github.com/anuragprasad95/loan_default_prediction_using_PCA/assets/3609255/015c27d7-68fb-4de4-975b-3b748eee3e11">
</samp>
