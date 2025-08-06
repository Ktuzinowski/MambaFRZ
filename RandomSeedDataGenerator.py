import random
import subprocess
import torch
import time

def main():
    py_script = "CombinedOperations.py"
    configuration_file = "training_configs/vgg11_bn_cifar100/similarity_guided_training.json"
    
    number_of_seeds_per_gpu = 1
    
    num_gpus = torch.cuda.device_count()
    random_seeds = list(set(random.randint(0, 10000) for _ in range(number_of_seeds_per_gpu * num_gpus)))
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
                "--configuration_file", configuration_file,
                "--seed", str(seed),
                "--device", f"cuda:{gpu}"
            ]
            
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