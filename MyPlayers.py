import MyModels
import MyUtils
import MyDatasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from Config import args
from torch.utils.data import DataLoader, TensorDataset



##############################################################################################################
##############################################################################################################
class Server():
    def __init__(self,  clients):
        self.clients = clients

        
    def aggregation(self):
        coeficients = 1 / torch.stack([client.num_samples for client in self.clients]).sum(dim=0) 
        summ = torch.stack([client.proto_logits * client.num_samples for client in self.clients]).sum(dim=0)
        self.ave_proto_logits = summ * coeficients 
        return self.ave_proto_logits
    
    def prompt_aggregation(self):
        num_all_samples = torch.stack(  [client.num_samples.sum(dim=0) for client in self.clients] ).sum(dim=0)   
        coeficients = torch.stack([client.num_samples.sum(dim=0) for client in self.clients]) / num_all_samples
        self.ave_prompts  = torch.stack(   [ client.model.ctx * coeficients[i]  for i, client in enumerate(self.clients)]   ).sum(dim=0)
        return self.ave_prompts
    
    
    def adapter_aggregation(self):
        # Step 1: Compute total number of samples across all clients
        num_all_samples = torch.stack([client.num_samples.sum(dim=0) for client in self.clients]).sum(dim=0)
    
        # Step 2: Compute coefficients for each client based on their sample count
        coefficients = torch.stack([client.num_samples.sum(dim=0) for client in self.clients]) / num_all_samples
    
        # Step 3: Aggregate classifier weights and biases
        weight_list = []
        bias_list = []
    
        for client in self.clients:
            head = client.model.classifier
            weight_list.append(head.weight.detach().clone())
            bias_list.append(head.bias.detach().clone())
    
        # Stack and weight the parameters
        stacked_weights = torch.stack(weight_list)  # shape: [num_clients, num_classes, feature_dim]
        stacked_biases = torch.stack(bias_list)     # shape: [num_clients, num_classes]
    
        # Reshape coefficients to broadcast properly
        coef = coefficients.view(-1, 1, 1)  # shape: [num_clients, 1, 1]
        aggregated_weight = (stacked_weights * coef).sum(dim=0)
        aggregated_bias = (stacked_biases * coefficients.view(-1, 1)).sum(dim=0)
    
        # Step 4: Assign aggregated parameters to all clients (or just one, depending on your design)
        for client in self.clients:
            client.model.classifier.weight.data.copy_(aggregated_weight)
            client.model.classifier.bias.data.copy_(aggregated_bias)

        


##############################################################################################################
##############################################################################################################
class Device():
    def __init__(self, ID, data, num_classes, name_classes):
        self.ID = ID
        self.data = data
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.num_samples = torch.bincount(self.data["train"]["label"], minlength=num_classes).to(args.device)


        ### Load the CLIP model for each setup 
        FM, processor, tokenizer = MyModels.load_clip_model()
        
        if "sl" in args.setup:
            self.model = MyModels.Prompt_Tuning_FM(FM, processor, tokenizer, self.num_classes, self.name_classes)
        elif "pr" in args.setup:
            self.model = MyModels.Prompt_Tuning_FM(FM, processor, tokenizer, self.num_classes, self.name_classes)
        elif "ad" in args.setup:
            self.model = MyModels.Linear_Probing_FM(FM, processor, tokenizer, num_classes, name_classes)
        else:
            raise ValueError("Name of the local model is not well-specified")


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.local_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
        self.loss_fn = torch.nn.functional.cross_entropy
        self.Loss = []
        self.Acc = []
        self.test_Acc = []
    def local_training(self):
        a,b, c = MyUtils.Train(self.model, self.data, self.optimizer, self.scheduler, self.loss_fn,
                               args.local_batch_size, args.local_epochs, args.device, args.debug)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c

    def local_distillation(self, teacher_knowledge, prototype=True):
        if prototype:
            teacher_knowledge = MyUtils.extend_proto_outputs_to_labels(self.data, teacher_knowledge)
        extended_data = MyDatasets.ddf({"student_model_input": self.data["train"]["image"], 
                                        "student_model_output":self.data["train"]["label"], 
                                        "teacher_knowledge": teacher_knowledge}
                                      )
        a, b, c = MyUtils.Distil(self.model, extended_data, self.data, self.optimizer, self.scheduler, self.loss_fn,
                                 args.local_batch_size, args.local_epochs, args.device, args.debug)
        self.Loss += a
        self.Acc += b
        self.test_Acc += c

    def cal_proto_logits(self, batch_size=64):
        images = self.data["train"]["image"]
        labels = self.data["train"]["label"]

        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size)

        all_logits = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(args.device)
                batch_labels = batch_labels.to(args.device)
                logits = self.model(batch_images)
                all_logits.append(logits)
                all_labels.append(batch_labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        unique_classes = sorted(set(labels.tolist()))
        num_classes = len(unique_classes)

        if "sift" in args.setup:
            predicted = torch.argmax(logits, dim=1)
            correct_mask = (predicted == labels)
            missing_classes = torch.tensor(
                [cls.item() for cls in labels.unique() if cls not in labels[correct_mask].unique()]
            ).to(args.device)
            missing_class_mask = torch.isin(labels, missing_classes)
            final_mask = correct_mask | missing_class_mask
            logits = logits[final_mask]
            labels = labels[final_mask]


        self.proto_logits = torch.empty((num_classes, num_classes), device=logits.device)

        
        for c in unique_classes:
            mask = (labels == c).to(logits.device)  
            category_logits = logits[mask].mean(dim=0)
            self.proto_logits[c] = category_logits



##############################################################################################################
##############################################################################################################



