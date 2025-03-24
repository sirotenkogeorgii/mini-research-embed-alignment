import os
import gc
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA


def check_directory(path: str) -> None:
    """Check if directory exists and create it if not"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)



def reduce_dim_single_path(config: DictConfig, dim: int, embeddings_path: str | Path) -> None:
    if dim > config.model.dim: 
        raise Exception(f"Target dim must be less the original number of dim: {dim} and {config.model.dim}.")
    if dim == config.model.dim: 
        return
    
    if not os.path.exists(embeddings_path):
        raise Exception(f"File with embeddings to reduce does not exist: {embeddings_path}.")

    save_path = embeddings_path.parent / f"{config.model.model_name}_{dim}.pth"

    data = torch.load(embeddings_path)
    embeddings = data["vectors"]
    
    
    if not save_path.exists():
        pca = PCA(n_components=dim, random_state=config.common.seed)
        reduced_emb = pca.fit_transform(embeddings)
        torch.save(
            {
                "dico": data["dico"],
                "vectors": torch.from_numpy(reduced_emb).float(),
            },
            save_path,
        )
        print(f"Saved {config.model.model_name}_{dim}.pth")

        del pca, reduced_emb
        gc.collect()


# def reduce_dim(args: DictConfig, models_config: Dict[str, Any]) -> None:
#     """Reduce dimensionality of embeddings using PCA"""
#     model_name = args.model.model_name
#     model_type = args.model.model_type
#     model_dim = args.model.dim
#     dataset_name = args.dataset.dataset_name
#     current_language = args.muse.language
    
#     # Set paths based on model type
#     if model_type.value == "vision_embeddings":
#         emb_root = Path(args.common.alias_emb_dir) / model_type.value / dataset_name
#         emb_path = emb_root / f"{model_name}_{model_dim}.pth"
        
#         # Target dimensionality based on alignment model
#         target_dim = args.muse.dim
#         if target_dim == model_dim:
#             logging.info(f"No dimension reduction needed for {model_name} - already at target dimension {target_dim}")
#             return
            
#         # Load embeddings
#         try:
#             embeddings = torch.load(emb_path)
#         except Exception as e:
#             logging.error(f"Error loading embeddings from {emb_path}: {str(e)}")
#             return
            
#         # Convert to numpy for PCA
#         embedding_keys = list(embeddings.keys())
#         embedding_values = np.stack([embeddings[k].numpy() if isinstance(embeddings[k], torch.Tensor) 
#                                    else embeddings[k] for k in embedding_keys])
        
#         # Apply PCA
#         logging.info(f"Reducing dimensionality from {model_dim} to {target_dim} using PCA")
#         pca = PCA(n_components=target_dim)
#         reduced_embeddings = pca.fit_transform(embedding_values)
        
#         # Convert back to dictionary
#         reduced_dict = {}
#         for i, key in enumerate(embedding_keys):
#             reduced_dict[key] = reduced_embeddings[i]
            
#         # Save reduced embeddings
#         reduced_path = emb_root / f"{model_name}_{target_dim}.pth"
#         torch.save(reduced_dict, reduced_path)
#         logging.info(f"Saved reduced embeddings to {reduced_path}")
        
#     else:  # Language embeddings
#         # For language models, we need to handle each language separately
#         emb_root = Path(args.common.alias_emb_dir) / model_type.value / current_language / "averaged_embeddings" / dataset_name
#         emb_path = emb_root / f"{model_name}_{model_dim}.pth"
        
#         # Target dimensionality based on alignment model
#         target_dim = args.muse.dim
#         if target_dim == model_dim:
#             logging.info(f"No dimension reduction needed for {model_name} - already at target dimension {target_dim}")
#             return
            
#         # Load embeddings
#         try:
#             embeddings = torch.load(emb_path)
#         except Exception as e:
#             logging.error(f"Error loading embeddings from {emb_path}: {str(e)}")
#             return
            
#         # Convert to numpy for PCA
#         embedding_keys = list(embeddings.keys())
#         embedding_values = np.stack([embeddings[k].numpy() if isinstance(embeddings[k], torch.Tensor) 
#                                    else embeddings[k] for k in embedding_keys])
        
#         # Apply PCA
#         logging.info(f"Reducing dimensionality from {model_dim} to {target_dim} using PCA")
#         pca = PCA(n_components=target_dim)
#         reduced_embeddings = pca.fit_transform(embedding_values)
        
#         # Convert back to dictionary
#         reduced_dict = {}
#         for i, key in enumerate(embedding_keys):
#             reduced_dict[key] = reduced_embeddings[i]
            
#         # Save reduced embeddings
#         reduced_path = emb_root / f"{model_name}_{target_dim}.pth"
#         torch.save(reduced_dict, reduced_path)
#         logging.info(f"Saved reduced embeddings to {reduced_path}") 