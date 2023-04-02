# Training models with multiple outputs.
# i.e X ~ y,z 

#%%
import pandas as pd
import io
import requests
from sklearn.model_selection import train_test_split

#%%
ee_dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"

def get_data() -> pd.DataFrame:
  response = requests.get(ee_dataset_url)
  file_bytes = io.BytesIO(response.content)
  return  pd.read_excel(file_bytes)

def get_x(df : pd.DataFrame) -> pd.DataFrame:
  return df[[x for x in df.columns.values if x[0] != 'Y']]

#%%
if __name__ == "__main__":
  
  #%%
  df = get_data()
  df.head(10)

  #%%
  train, test = train_test_split(df, test_size= 0.2)

# %%
