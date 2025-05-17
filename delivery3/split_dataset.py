import json
import random

input_dataset_path = "AlpacaStyle_DatasetCombined.json"  
output_testset_path = "CSE4078S25_Grp1_testset_AlpacaStyle.json"
num_test_samples = 1000  

with open(input_dataset_path, "r", encoding="utf-8") as f:
    full_data = json.load(f)

random.shuffle(full_data)
test_data = full_data[:num_test_samples]

with open(output_testset_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

train_data = full_data[num_test_samples:]
with open("CSE4078S25_Grp1_train_AlpacaStyle.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"Test dataset saved to {output_testset_path} with ({len(test_data)} samples)")
print(f"Training dataset saved without the test dataset with ({len(train_data)} samples)")
