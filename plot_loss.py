import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from Config import Config

# read log files

df = pd.read_csv(os.path.join(Config.LOG_DIR, 'training_log.csv'))
print(df.columns)
sns.set()

plt.figure(figsize=(10, 8))
plt.plot(df['Epoch'], df['Loss'])
plt.title('Training Loss over Epochs')
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(df['Epoch'], df['Accuracy'])
plt.title('Training Accuracy over Epochs')
plt.show()
