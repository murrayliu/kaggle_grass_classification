import numpy as np

# Sample NumPy array
data = [['test', 'tset'],
                 [4, 5],
                 [7, 8]]

# Specify the CSV file path
csv_file_path = 'output.csv'

# Column headers
headers = ['file', 'species']

# Writing to the CSV file with headers
np.savetxt(csv_file_path, data, delimiter=',', fmt='%s', header=','.join(headers), comments='')

print(f'Data has been written to {csv_file_path}')
