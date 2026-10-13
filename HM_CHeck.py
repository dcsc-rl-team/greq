import pickle
import numpy as np

file_path = "./Data/Uniform_Sampling/Pre_Cond_QL_Garnet_Random/0_999/Pre-Cond_Q-Learning_(Randoms)_0.999_Trial_1_results.pkl"

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("--- Root Keys ---")
print(data.keys())

print("\n--- Metrics Available ---")
print(data['metrics'].keys())

# Let's inspect the array shape of your target metric
values = np.array(data['metrics']['value_norm'])
print(f"\nShape of 'value_norm' array: {values.shape}")