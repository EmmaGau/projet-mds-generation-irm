import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('checkpoints/20240118_162705/progress.csv')

# Plot the data
fig, ax = plt.subplots()
ax.plot([10*i for i in range(len(df['loss']))], df['loss'], label='loss')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.legend()

# Save the figure
fig.savefig('checkpoints/20240118_162705/loss.png')