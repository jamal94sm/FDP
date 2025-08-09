import numpy as np
import transformers
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import args
import pandas as pd





##############################################################################################################
def load_clip_model():
    model_name = args.Foundation_model
    model = transformers.CLIPModel.from_pretrained(model_name)
    processor = transformers.CLIPProcessor.from_pretrained(model_name, use_fast=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if "BN" in args.setup: 
        print("Unfreeze LayerNorm layers in the image encoder")

        # Unfreeze LayerNorm layers in the image encoder
        for module in model.vision_model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.train()  # Set to training mode
                for param in module.parameters():
                    param.requires_grad = True
                    
    return model, processor, tokenizer
##############################################################################################################
##############################################################################################################
class LLM(torch.nn.Module):
    def __init__(self):
        super(LLM, self).__init__()
        model_name = "distilgpt2"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    def generate_response(self, max_length=100):
        inputs = self.tokenizer(args.prompt_template, return_tensors="pt").to(args.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
##############################################################################################################
##############################################################################################################
class Prompt_Tuning_FM(torch.nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes):
        super(Prompt_Tuning_FM, self).__init__()
        self.FM = FM.to(args.device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_classes = num_classes
        self.name_classes = name_classes

        for param in self.FM.parameters():
            param.requires_grad = False

        self.embedding_lookup_table = self.FM.text_model.embeddings

        self.ctx = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(args.num_prompts, self.FM.config.text_config.hidden_size, device=args.device),
                std=0.02
            )
        )

        self.build_token_embeds()
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value, device=args.device))

        self.Loss = []
        self.Acc = []
        self.test_Acc = []

    def build_token_embeds(self):
        self.tokens = self.tokenizer(
            self.name_classes,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )["input_ids"].to(args.device)

        with torch.no_grad():
            self.token_embeds = self.embedding_lookup_table(self.tokens)

    def prepare_prompt(self):
        ctx = torch.unsqueeze(self.ctx, dim=0)
        ctx = torch.broadcast_to(ctx, [self.num_classes, ctx.shape[-2], ctx.shape[-1]])
        self.prompts = torch.cat([self.token_embeds[:, :1, :], ctx, self.token_embeds[:, 1:, :]], dim=1)

    def customize_images(self, imgs):
        processed = self.processor(images=imgs, return_tensors="pt", padding=True)['pixel_values']
        return processed.to(args.device)

    def __call__(self, images):
        self.prepare_prompt()

        output = self.FM.text_model.encoder(inputs_embeds=self.prompts)
        
        last_hidden_state = self.FM.text_model.final_layer_norm(output[0])
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=args.device),
            self.tokens.argmax(dim=-1)
        ]
        
        text_rep = self.FM.text_projection(pooled_output)

        images = self.customize_images(images)
        img_rep = self.FM.get_image_features(images)

        img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)
        text_rep = text_rep / text_rep.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_rep @ text_rep.t()
        
        return logits




##############################################################################################################
##############################################################################################################


class Linear_Probing_FM(torch.nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes):
        super(Linear_Probing_FM, self).__init__()
        self.FM = FM.to(args.device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_classes = num_classes
        self.name_classes = name_classes

        # Freeze all parameters in the CLIP model
        for param in self.FM.parameters():
            param.requires_grad = False

        # Classification head for linear probing
        self.classifier = torch.nn.Linear(self.FM.config.projection_dim, num_classes).to(args.device)

        self.logit_scale = torch.nn.Parameter(
            torch.tensor(self.FM.config.logit_scale_init_value, device=args.device)
        )

        self.Loss = []
        self.Acc = []
        self.test_Acc = []

    def customize_images(self, imgs):
        processed = self.processor(images=imgs, return_tensors="pt", padding=True)['pixel_values']
        return processed.to(args.device)

    def forward(self, images):
        # Preprocess and encode images using frozen CLIP
        images = self.customize_images(images)
        img_rep = self.FM.get_image_features(images)

        # Normalize image representations
        img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)

        # Pass through trainable classifier head
        logits = self.classifier(img_rep)

        return logits








class Prompt_Tuning_FM2(torch.nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes):
      super(Prompt_Tuning_FM2, self).__init__()
      self.FM = FM
      self.tokenizer = tokenizer
      self.processor = processor
      for param in self.FM.parameters(): param.requires_grad = False
      self.num_classes = num_classes
      self.name_classes = name_classes
      self.embedding_lookup_table = self.FM.text_model.embeddings
    
      self.ctx = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(args.num_prompts, self.FM.config.text_config.hidden_size), std=0.02))
      self.build_token_embeds()
      self.logit_scale = torch.nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value))
      self.Loss = []
      self.Acc = []
      self.test_Acc = []
      
    def build_token_embeds(self):
      self.tokens = self.tokenizer(self.name_classes, add_special_tokens=True, padding=True, return_tensors="pt")["input_ids"]
      with torch.no_grad():
          self.token_embeds = self.embedding_lookup_table(self.tokens)
          
    def prepare_prompt(self):
      ctx = torch.unsqueeze(self.ctx, dim=0)
      ctx = torch.broadcast_to(ctx, [self.num_classes, ctx.shape[-2], ctx.shape[-1]])
      self.prompts = torch.cat([self.token_embeds[:, :1, :], ctx, self.token_embeds[:, 1:, :]], dim=1)
      
    def customize_images(self, imgs):
      return self.processor( images=imgs, return_tensors="pt", padding=True)['pixel_values']
  
    def __call__(self, images):
      self.prepare_prompt()
      output = self.FM.text_model.encoder(inputs_embeds = self.prompts)
      last_hidden_state = self.FM.text_model.final_layer_norm(output[0])
      pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]),  self.tokens.argmax(dim=-1)]
      text_rep = self.FM.text_projection(pooled_output)
    
      images = self.customize_images(images)
      img_rep = self.FM.get_image_features(images)
    
      img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)
      text_rep = text_rep / text_rep.norm(p=2, dim=-1, keepdim=True)
    
      logit_scale = self.logit_scale.exp()
      logits = logit_scale * img_rep @ text_rep.t()
      soft_labels = torch.nn.functional.softmax(logits, dim=1)
      return soft_labels

    