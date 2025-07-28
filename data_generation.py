import torch
import os
import pickle
import re

def generate_formatted_data(frz_predictor_type, file_list, dataset_file, folder_location, subfolder, total_count):
  counter = 0

  # Pattern for extracting Layer Name, Epoch, and Seed
  pattern = r"layer_([^_]*)_epoch_(\d*)_(\d*)"

  for filename in file_list:
    if filename.endswith('.pkl'):
      counter += 1
      print(filename, f"{counter}/{total_count}")
      with open(os.path.join(folder_location, filename), "rb") as f:
        tensor_data = pickle.load(f)
        freeze_input = tensor_data[0]
        
        if frz_predictor_type == "smartfrz":
            for index, weight in enumerate(tensor_data):
                if index == 0:
                    continue
                freeze_input = torch.cat((freeze_input, weight), 1)
        elif frz_predictor_type == "mambafrz":
            for index, weight in enumerate(tensor_data):
                if index == 0:
                    continue
                freeze_input = torch.cat((freeze_input, weight), 0)

        match_for_info = re.match(pattern, filename)
        if match_for_info:
          layer_name = match_for_info.group(1)
          epoch = match_for_info.group(2)
          seed = match_for_info.group(3)
          output_response = (freeze_input, layer_name, epoch, seed)
        else:
          raise ValueError(f"No match within {filename}, does not match and hence fails")

        dataset_file['data'].append(output_response)

        if subfolder == 'frz':
          dataset_file['labels'].append(1)
        else:
          dataset_file['labels'].append(0)

def generate_compressed_dataset(root_dir, total_count, frz_predictor_type):
  compressed_dataset_file = {
    'data': [],
    'labels': []
  }
  
  for subfolder in ['frz', 'nofrz']:
    pickle_folder_location = os.path.join(root_dir, subfolder)
    file_list = os.listdir(pickle_folder_location)
    generate_formatted_data(frz_predictor_type, file_list, compressed_dataset_file, pickle_folder_location, subfolder, total_count)
  with open(f"{root_dir}/compressed_dataset_{frz_predictor_type}.pkl", "wb") as f:
    pickle.dump(compressed_dataset_file, f)