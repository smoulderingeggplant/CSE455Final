input_file = "Multi+NN151515Small"
input_file_path = "NNOutputs/" + input_file + ".txt"

# Read the data from the file
with open(input_file_path, 'r') as file:
    data = file.readlines()

# Extract file names and accuracy values separately
file_names = []
accuracies = []
for line in data:
    if line.startswith("Testing CSV file"):
        file_name = line.split()[-1].split('/')[-1].split('.')[0]
        file_names.append('"' + file_name + '"')
    elif line.startswith("Accuracy"):
        accuracies.append(line.split(":")[1].strip())

# Organize the data
output_data = ""
for file_name in file_names:
    output_data += file_name + ",\n"

output_data += "\n"

for accuracy in accuracies:
    output_data += accuracy + ",\n"

# Write the output to a new file
output_file = "NNOutputs/" + input_file + "Organized.txt"
with open(output_file, 'w') as file:
    file.write(output_data)

print("Output written to", output_file)