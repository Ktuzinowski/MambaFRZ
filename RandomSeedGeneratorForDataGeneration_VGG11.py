import random
import subprocess
import torch
import time

def main():
    # --- CONFIGURATION ----------------
    py_script = "CombinedOperation_VGG11.py"
    name_of_experiment = "mambafrz_vgg11_MambaFRZ_validation_12_seeds"
    fully_trained_reference_model = f"test_model_weights/best_model_VGG11.pt"
    window_size = 30
    frz_predictor_path = "/home/idies/workspace/Storage/ktuzinows1/persistent/MambaFRZ/mambafrz_vgg11_data_generation_12_seeds/training_data/context_window_30/mambafrz_architecture_change_test/mambafrz_trained_8.pth"
    number_of_cnn_layers = 0
    frz_from_frz_predictor = True
    use_linear_restriction = True
    similarity_guided_training = False
    save_fully_trained_ref_model = False
    num_epochs = 200
    frz_predictor_model_name = "mambafrz"
    # ----------------------------------
    
    number_of_seeds_per_gpu = 1
    
    num_gpus = torch.cuda.device_count()
    # random_seeds = list(set(random.randint(0, 10000) for _ in range(number_of_seeds_per_gpu * num_gpus)))
    random_seeds = [2200]
    max_procs = num_gpus
    
    assert len(random_seeds) == number_of_seeds_per_gpu * num_gpus
    
    # Initialize your "free GPU" pool and process list
    free_gpus = list(range(max_procs))
    processes = [] # will hold tuples (Popen_obj, gpu_id, seed)
    
    while random_seeds or processes:
        # 1) Launch as many new procs as we can (GPUs free & seeds left)
        while free_gpus and random_seeds:
            gpu = free_gpus.pop(0)
            seed = random_seeds.pop(0)
            
            cmd = [
                "python", py_script,
                "--name_of_experiment", name_of_experiment,
                "--window_size", str(window_size),
                "--frz_predictor_path", frz_predictor_path,
                "--seed", str(seed),
                "--fully_trained_reference_model", fully_trained_reference_model,
                "--number_of_cnn_layers", str(number_of_cnn_layers), 
                "--cuda_device", f"cuda:{gpu}",
                "--epochs", str(num_epochs),
                "--frz_predictor_model_name", frz_predictor_model_name
            ]
            if frz_from_frz_predictor:
                cmd.append("--frz_from_frz_predictor")
            if use_linear_restriction:
                cmd.append("--use_linear_restriction")
            if similarity_guided_training:
                cmd.append("--similarity_guided_training")
            if save_fully_trained_ref_model:
                cmd.append("--save_fully_trained_ref_model")
            
            print(f"→ Launching seed={seed} on GPU {gpu}")
            p = subprocess.Popen(cmd)
            processes.append((p, gpu, seed))
        
        # 2) Poll running processes and reap any that have finished
        for (p, gpu, seed) in processes[:]:
            if p.poll() is not None: # process has exited
                ret = p.returncode
                print(f"✔ Seed={seed} on GPU {gpu} finished (code={ret})")
                processes.remove((p, gpu, seed))
                free_gpus.append(gpu) # free up that GPU
        
        # 3) If we're at capacity and nothing freed, wait a bit
        if len(processes) >= max_procs:
            time.sleep(1.0)
    print("All done!")
            
if __name__ == "__main__":
    main()