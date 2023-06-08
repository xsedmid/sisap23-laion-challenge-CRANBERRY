import h5py
import subprocess

# Print the current file path
print(f"The current running file: {__file__}")

result_file_path = 'test.h5'

java_output = subprocess.check_output(['java', 'TestJavaClass', '7', '3'], universal_newlines=True)

# Convert output to an integer
result = int(java_output.strip())

print(f"The result from Java: {result}")

# Store the result into a h5 file
with h5py.File(result_file_path, 'w') as f:
    f.create_dataset('result', data=result)

# Read and print the result from the h5 file
with h5py.File(result_file_path, 'r') as f:
    result = f['result'][()]
    print(f"The result from h5: {result}")
