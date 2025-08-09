import torch
import transformers
import numpy as np
import random
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import MyDatasets
import MyModels
import MyPlayers
import MyUtils
import torchvision
import time
import json
import os
import gc
from Config import args 
import time
import psutil










def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)
    transformers.set_seed(seed)


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()    




##################################################################################
##################################################################################
def main():

    
    device = torch.device(args.device)
    print(f'Device: {device}')




    # ===================== Client and Server Setup =====================
    clients = [
        MyPlayers.Device(
            id,
            distributed_dataset[id],
            num_classes,
            name_classes
        ) for id in range(args.num_clients)
    ]

    server = MyPlayers.Server(clients)




    # ===================== Training Rounds =====================




    for client in clients:
        client.local_training()



    for round_idx in range(args.rounds):
        print("=" * 20, f" Round {round_idx + 1}/{args.rounds} ", "=" * 20)
        
        if "sl" in args.setup:
            for client in clients:
                client.cal_proto_logits()
            agg = server.aggregation()
            
            
            for client in clients:
                print(f"Distillation process of client {client.ID}:")
                client.local_distillation(agg, prototype=True)
            
            
        elif "pr" in args.setup: 
            prompt_agg = server.prompt_aggregation()
            
            for client in clients:
                print(f"Prompt Tuning process of client {client.ID}:")
                client.model.ctx.data.copy_(prompt_agg.to(args.device))
                client.local_training()
            

        elif "ad" in args.setup:
            server.adapter_aggregation()
            
            for client in clients:
                client.local_training()
        else:
            raise ValueError("This is a custom error message.")
        







        
    # ===================== Save Results =====================
    avg_test_Acc = np.mean([client.test_Acc for client in clients], axis=0)
    MyUtils.save_as_json(avg_test_Acc, args, file_name= args.output_name + "accuracy_"+args.setup)







        

##################################################################################
##################################################################################
if __name__ == "__main__":
    
    
    set_seed(42)



    # ===================== Dataset and Model Loading =====================
    Dataset, num_classes, name_classes = MyDatasets.load_data_from_Huggingface()



    # ===================== Data Distribution =====================
    distributed_dataset, num_samples = MyDatasets.data_distributing(Dataset, num_classes)
    print("\n ]data distribution of devices: \n", num_samples)



    # ===================== Run for each configuration =====================
    # ft: clip is fine-tuned --- mean: average of descriptions' embedding is used for refrence
    # M: multiple descriptions --- sift: only true_labeled soft labels be shared with the server
    configurations = [
        #{"setup": "clinetFM_sl_yn"}, # sl: soft label
        #{"setup": "clinetFM_pr_yn"}, # pr: promt
        {"setup": "clinetFM_ad_yn"}, # ad: adaptor
    ]

    for config in configurations:
        args.setup = config["setup"]
        

            
        separator = "=" * 40
        print(f"\n{separator} Running configuration: {args.setup} {separator}")
    
        
        
        main()
        
        clean_memory()
        print(f"{separator} Simulation is over for configuration {args.setup} {separator}\n")









    
    
    
    # ===================== Data Loading and Plot =====================
    results_dir = "results"  # Directory containing your JSON files    
    stored_arrays = []  # Collect all 'stored' arrays
    names = []
    for file in os.listdir(results_dir):
        if file.endswith(".json") and file.startswith(args.output_name):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                if "stored" in data:
                    arr = np.array(data["stored"])
                    stored_arrays.append(arr) 
                if "setup" in data:
                    names.append(data["setup"])

    MyUtils.plot(stored_arrays, names)

    

    #MyUtils.play_alert_sound()
    







