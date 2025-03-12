import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# import clip
import numpy as np
import requests
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


class VMEmbedding:
    def __init__(self, args: DictConfig, image_labels: List[str]):
        self.config: DictConfig = args
        self.model_id: str = args.model.model_id
        self.bs = 32
        self.model_name: str = args.model.model_name
        self.model_dim = args.model.dim
        self.dataset_name: str = args.dataset.dataset_name
        self.image_dir: str = args.dataset.image_dir
        self.image_labels: List[str] = image_labels
        self.alias_emb_dir: Path = (
                Path(args.common.alias_emb_dir) / args.model.model_type.value / self.dataset_name
        )
        
        # Create directory
        os.makedirs(self.alias_emb_dir, exist_ok=True)
        
        # HuggingFace endpoint info
        self.use_hf_endpoint: bool = bool(args.hf_endpoint.api_key and args.hf_endpoint.vm_endpoint)
        self.hf_api_key: str = args.hf_endpoint.api_key
        self.hf_endpoint_url: str = args.hf_endpoint.vm_endpoint
        self.max_concurrent_requests: int = args.hf_endpoint.max_concurrent_requests
        self.timeout: int = args.hf_endpoint.timeout
        
        self.device = "cuda" if torch.cuda.is_available() and not self.use_hf_endpoint else "cpu"

    def get_vm_layer_representations(self) -> None:
        """Process vision embeddings using either local model or HF endpoint"""
        save_file_path = self.alias_emb_dir / f"{self.model_name}_{self.model_dim}.pth"
        
        if self.use_hf_endpoint:
            self._process_with_hf_endpoint(save_file_path)
        else:
            self._process_with_local_model(save_file_path)
            
        print(f"Vision embeddings successfully extracted for {self.dataset_name}!")

    def _process_with_local_model(self, save_file_path: Path) -> None:
        """Process vision embeddings with local CLIP model"""
        # Load CLIP model
        model, preprocess = clip.load(self.model_id, device=self.device)
        model.eval()
        
        # Process images batch by batch
        all_embeddings = {}
        
        for batch_start in tqdm(range(0, len(self.image_labels), self.bs), 
                              desc=f"Processing {self.model_name} embeddings"):
            batch_end = min(batch_start + self.bs, len(self.image_labels))
            batch_labels = self.image_labels[batch_start:batch_end]
            
            processed_images = []
            for label in batch_labels:
                image_path = os.path.join(self.image_dir, f"{label}.jpg")
                try:
                    image = Image.open(image_path).convert("RGB")
                    processed_images.append(preprocess(image))
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {str(e)}")
                    # Use zero tensor as fallback
                    processed_images.append(torch.zeros(3, 224, 224))
            
            # Stack images and process
            image_tensor = torch.stack(processed_images).to(self.device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Store embeddings
            for i, label in enumerate(batch_labels):
                all_embeddings[label] = image_features[i].cpu().numpy()
        
        # Save embeddings
        torch.save(all_embeddings, save_file_path)
    
    def _process_with_hf_endpoint(self, save_file_path: Path) -> None:
        """Process vision embeddings with HF Inference Endpoint"""
        all_embeddings = {}
        
        for batch_start in tqdm(range(0, len(self.image_labels), self.bs), 
                              desc=f"Processing {self.model_name} embeddings via HF Endpoint"):
            batch_end = min(batch_start + self.bs, len(self.image_labels))
            batch_labels = self.image_labels[batch_start:batch_end]
            
            # Load and encode images
            image_paths = [os.path.join(self.image_dir, f"{label}.jpg") for label in batch_labels]
            
            # Get embeddings from HF endpoint
            embeddings = self._get_embeddings_from_hf_endpoint(image_paths)
            
            # Store embeddings
            for i, label in enumerate(batch_labels):
                if i < len(embeddings):  # Safety check
                    all_embeddings[label] = embeddings[i]
        
        # Save embeddings
        torch.save(all_embeddings, save_file_path)
    
    def _get_embeddings_from_hf_endpoint(self, image_paths: List[str]) -> List[np.ndarray]:
        """Get embeddings from HuggingFace Inference Endpoint"""
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
        }
        
        # Prepare images for API request
        files = []
        for i, img_path in enumerate(image_paths):
            try:
                # Check if file exists
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        files.append(
                            (f'files', (f'image_{i}.jpg', img_file.read(), 'image/jpeg'))
                        )
                else:
                    logging.warning(f"Image file not found: {img_path}")
            except Exception as e:
                logging.error(f"Error reading image {img_path}: {str(e)}")
        
        try:
            response = requests.post(
                self.hf_endpoint_url,
                headers=headers,
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, dict) and "error" in result:
                raise ValueError(f"Error from HF Endpoint: {result['error']}")
                
            # Return embeddings
            return [np.array(emb) for emb in result]
            
        except Exception as e:
            logging.error(f"Error calling HF Endpoint: {str(e)}")
            # Return empty embeddings as fallback
            return [np.zeros(self.model_dim) for _ in image_paths] 