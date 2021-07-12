import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
df=pd.read_csv("fake_job_postings.csv")
for column in df.columns:
    df[column]=df[column].fillna(f'missing_{column}')
seri=df.groupby('telecommuting').mean()['fraudulent']
plt.title('telecommuting')
plt.xlabel('index')
plt.ylabel('percentage of fraudulent')
plt.bar(np.array(seri.index,dtype='str'),seri)
plt.show()
seri=df.groupby('has_company_logo').mean()['fraudulent']
plt.title('has_company_logo')
plt.xlabel('index')
plt.ylabel('percentage of fraudulent')
plt.bar(np.array(seri.index,dtype='str'),seri)
plt.show()
seri=df.groupby('has_questions').mean()['fraudulent']
plt.title('has_questions')
plt.xlabel('index')
plt.ylabel('percentage of fraudulent')
plt.bar(np.array(seri.index,dtype='str'),seri)
plt.show()
seri=df.groupby('employment_type').mean()['fraudulent']
plt.figure(figsize=(20,8))
plt.title('employment_type')
plt.xlabel('index')
plt.ylabel('percentage of fraudulent')

plt.bar(np.array(seri.index,dtype='str'),seri)
plt.show()
seri=df.groupby('required_education').mean()['fraudulent']
for k,ind in enumerate(np.array(seri.index,dtype='str')):
   print(f'index {k+1}: {ind}',end=", ")
plt.figure(figsize=(20,8))
plt.title('required_education')
plt.xlabel('index')
plt.ylabel('percentage of fraudulent')
plt.xticks(np.arange(1,len(seri)+1,1))
plt.bar(np.arange(1,len(seri)+1,1),seri)
plt.show()
