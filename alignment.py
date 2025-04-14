import os
import hydra
import json
import logging
import re
import random
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from collections import OrderedDict

from dataclasses import dataclass, replace, fields

from src.config import MODEL_CONFIGS, RunConfig, MuseConfig
from src.procrustes import MuseExp
from src.utils.utils_helper import reduce_dim_single_path

from MUSE.src.evaluation import Evaluator
from MUSE.src.models import build_model
from MUSE.src.trainer import Trainer
from MUSE.src.utils import initialize_exp

import tempfile
import shutil
import torch



def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _train(configs: MuseConfig) -> dict:
    params = configs
    # check parameters
    # assert not params.cuda or torch.cuda.is_available()
    assert params.dico_train in ["identical_char", "default"] or os.path.isfile(
        params.dico_train
    )
    assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
    assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
    assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size

    print(params.src_emb, params.emb_dim)

    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    assert params.dico_eval == "default" or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)

    trainer.load_training_dico(params.dico_train)

    VALIDATION_METRIC_SUP = "precision_at_1-csls_knn_100"
    VALIDATION_METRIC_UNSUP = "mean_cosine-csls_knn_100-S2T-10000"

    # define the validation metric
    VALIDATION_METRIC = (
        VALIDATION_METRIC_UNSUP
        if params.dico_train == "identical_char"
        else VALIDATION_METRIC_SUP
    )
    logger.info("Validation metric: %s" % VALIDATION_METRIC)

    """
    Learning loop for Procrustes Iterative Learning
    """

    n_iter = 0
    # logger.info("Starting iteration %i..." % n_iter)

    # # build a dictionary from aligned embeddings (unless
    # # it is the first iteration and we use the init one)
    # if n_iter > 0 or not hasattr(trainer, "dico"):
    #     trainer.build_dictionary()

    # # apply the Procrustes solution
    # # for _ in range(5):
    # trainer.procrustes()

    for n_iter in range(params.n_refinement + 1):

        logger.info('Starting iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings (unless
        # it is the first iteration and we use the init one)
        if n_iter > 0 or not hasattr(trainer, 'dico'):
            trainer.build_dictionary()

        # apply the Procrustes solution
        trainer.procrustes()

        # embeddings evaluation
        to_log = OrderedDict({'n_iter': n_iter})
        evaluator.all_eval(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of iteration %i.\n\n' % n_iter)

    return trainer

def muse_supervised(configs: MuseConfig) -> dict:
    params = configs

    trainer = _train(params)
    evaluator = Evaluator(trainer)


    print(f"[DEBUG] Evaluation")
    # embeddings evaluation
    to_log = OrderedDict({"n_iter": 0})
    best_mapping_path = os.path.join(params.exp_path, f'{params.lm}_{params.vm}_dim{params.emb_dim}_best_mapping.pth')
    evaluator.load_mapping(best_mapping_path)
    evaluator.all_eval(to_log)

    topk_nn = evaluator.get_topk_nn(k=params.topk)
    evaluator.save_aligned_embs_word2id(params.save_embeddings_dir, params.save_word2id_dir, f"{params.lm}_{params.vm}_dim{params.emb_dim}")
    

    result_metrics = {}
    for metrics, value in to_log.items():
        if "mean_cosine" in metrics:
            result_metrics[metrics] = value
        else:
            re_catch = re.search(r'precision_at_(\d+)', metrics)
            if re_catch is None: continue
            k = int(re_catch.group(1))

            if "csls" in metrics:
                metrics_name = "CSLS"
            elif "nn" in metrics:
                metrics_name = "NN"
            result_metrics[f"P@{k}-{metrics_name}"] = value


    return result_metrics, topk_nn



def muse_supervised_kfolds(configs: MuseConfig) -> dict:
    params = configs

    def __merge_folds(fold_files, exclude_index):
        temp_train = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        
        for i, fold_path in enumerate(fold_files):
            if i == exclude_index:
                val_file = fold_path
            else:
                with open(fold_path, 'r') as f:
                    shutil.copyfileobj(f, temp_train)
        
        temp_train.flush()
        temp_train.seek(0)
        
        return temp_train.name, val_file

    
    folds_results = []
    folds_txts = [f"{params.kfolds_dir}/{file_name}" for file_name in sorted(os.listdir(params.kfolds_dir))]
    for fold_idx in range(len(folds_txts)):
        print(f"--------------------------- FOLD {fold_idx + 1} ---------------------------")
        train_path, val_path = __merge_folds(folds_txts, exclude_index=fold_idx)

        params.dico_train = train_path
        params.dico_eval = val_path

        trainer = _train(params)
        evaluator = Evaluator(trainer)

        print(f"[DEBUG] Evaluation")
        # embeddings evaluation
        to_log = OrderedDict({"n_iter": 0})
        best_mapping_path = os.path.join(params.exp_path, f'{params.lm}_{params.vm}_dim{params.emb_dim}_best_mapping.pth')
        evaluator.load_mapping(best_mapping_path)
        evaluator.all_eval(to_log)

        # topk_nn = evaluator.get_topk_nn(k=params.topk)
        # evaluator.save_aligned_embs_word2id(params.save_embeddings_dir, params.save_word2id_dir, f"{params.lm}_{params.vm}_dim{params.emb_dim}")

        result_metrics = {}
        for metrics, value in to_log.items():
            if "mean_cosine" in metrics:
                result_metrics[metrics] = value
            else:
                re_catch = re.search(r'precision_at_(\d+)', metrics)
                if re_catch is None: continue
                k = int(re_catch.group(1))

                if "csls" in metrics:
                    metrics_name = "CSLS"
                elif "nn" in metrics:
                    metrics_name = "NN"
                result_metrics[f"P@{k}-{metrics_name}"] = value

        folds_results.append(result_metrics)
        params.src_mean = None
        params.tgt_mean = None
        os.remove(train_path)
        print()
    
    print(f"--------------------------- ALL DATA ---------------------------")

    params.dico_train = params.all_data_path
    params.dico_eval = params.all_data_path

    trainer = _train(params)
    evaluator = Evaluator(trainer)

    print(f"[DEBUG] Evaluation")
    # embeddings evaluation
    to_log = OrderedDict({"n_iter": 0})
    best_mapping_path = os.path.join(params.exp_path, f'{params.lm}_{params.vm}_dim{params.emb_dim}_best_mapping.pth')
    evaluator.load_mapping(best_mapping_path)
    evaluator.all_eval(to_log)

    # topk_nn = evaluator.get_topk_nn(k=params.topk)
    # evaluator.save_aligned_embs_word2id(params.save_embeddings_dir, params.save_word2id_dir, f"{params.lm}_{params.vm}_dim{params.emb_dim}")

    result_metrics = {}
    for metrics, value in to_log.items():
        if "mean_cosine" in metrics:
            result_metrics[metrics] = value
        else:
            re_catch = re.search(r'precision_at_(\d+)', metrics)
            if re_catch is None: continue
            k = int(re_catch.group(1))

            if "csls" in metrics:
                metrics_name = "CSLS"
            elif "nn" in metrics:
                metrics_name = "NN"
            result_metrics[f"P@{k}-{metrics_name}"] = value

    folds_results.append(result_metrics)
    params.src_mean = None
    params.tgt_mean = None
    print()

    return folds_results, []



def reduce_dim_single_path(embeddings_path: str | Path, dim: int) -> str | Path:
    """
    Reduce the dimensionality of embeddings using PCA.
    
    Args:
        embeddings_path: Path to the embeddings file
        dim: Target dimension
        
    Returns:
        Path to the reduced embeddings file
    """
    from pathlib import Path
    import torch
    import gc
    from sklearn.decomposition import PCA
    
    embeddings_path = Path(embeddings_path)
    
    if not embeddings_path.exists():
        raise Exception(f"File with embeddings to reduce does not exist: {str(embeddings_path)}.")

    # Extract model name from filename (assuming format modelname_dimension.pth)
    model_name = "_".join(embeddings_path.stem.split('_')[:-1])
    
    # Create save path in same directory
    save_path = embeddings_path.parent / f"{model_name}_{dim}.pth"
    
    # If file already exists, return its path
    if save_path.exists():
        print(f"Using existing reduced embeddings: {save_path}")
        return str(save_path)
    
    # Load data
    data = torch.load(embeddings_path)
    embeddings = data["vectors"]
    
    # Apply PCA
    pca = PCA(n_components=dim, random_state=42)  # Using fixed seed for reproducibility
    reduced_emb = pca.fit_transform(embeddings)
    
    # Save reduced embeddings
    torch.save(
        {
            "dico": data["dico"],
            "vectors": torch.from_numpy(reduced_emb).float(),
        },
        save_path,
    )
    print(f"Saved reduced embeddings to {save_path}")

    # Clean up
    del pca, reduced_emb
    gc.collect()
    
    return str(save_path)


def reduce_dimensions(args: MuseConfig):
    min_dim = args.emb_dim
    source_file = args.src_emb
    print(f'[DEBUG](reduce_dimensions): {source_file}')
    target_file = args.tgt_emb
    
    # Path(args.common.embeddings_dataset_root).mkdir(parents=True, exist_ok=True)
    for model_name, filename in [(args.vm, source_file), (args.lm, target_file)]:
        if not os.path.exists(filename):
            model_dim = MODEL_CONFIGS[model_name].dim
            file_dir = "/".join(filename.split("/")[:-1])
            default_filename = f"{file_dir}/{model_name}_{model_dim}.pth"
            reduce_dim_single_path(default_filename, min_dim)


def resolve_config_paths(config_obj):
    """
    Ensure all path-like strings in the config object are fully resolved.
    
    Args:
        config_obj: Configuration object with path attributes
        
    Returns:
        The same configuration object with resolved paths
    """
    # List of common path attributes that might need resolution
    path_attributes = [
        'src_emb', 'tgt_emb', 'dico_train', 'dico_eval', 
        'result_metrics_save_dir', 'full_dict_path'
    ]
    
    for attr in path_attributes:
        if hasattr(config_obj, attr):
            value = getattr(config_obj, attr)
            if isinstance(value, str) and ('${' in value or '~' in value):
                resolved_path = os.path.expandvars(os.path.expanduser(str(value)))
                setattr(config_obj, attr, resolved_path)
                print(f"[DEBUG] Resolved {attr}: {resolved_path}")
    
    return config_obj


def create_sampled_dictionaries(
    full_dict_path: str, 
    train_size: int, 
    seed: int,
    temp_dir: str = "/tmp"
) -> Tuple[str, str]:
    """
    Create train and eval dictionaries by sampling from a complete dictionary file.
    
    Args:
        full_dict_path: Path to the complete dictionary file
        train_size: Number of pairs to include in the training dictionary
        seed: Random seed for reproducibility
        temp_dir: Directory to save temporary dictionary files
        
    Returns:
        Tuple of (train_dict_path, eval_dict_path)
    """
    # Set random seed
    random.seed(seed)
    
    # Read all pairs from the dictionary
    with open(full_dict_path, 'r', encoding='utf-8') as f:
        all_pairs = [line.strip() for line in f if line.strip()]
    
    # Shuffle the pairs
    random.shuffle(all_pairs)
    
    # Ensure train_size is valid
    if train_size >= len(all_pairs):
        raise ValueError(f"train_size ({train_size}) must be less than the number of pairs in the dictionary ({len(all_pairs)})")
    
    # Split into train and eval sets
    train_pairs = all_pairs[:train_size]
    eval_pairs = all_pairs[train_size:]
    
    # Create file paths
    os.makedirs(temp_dir, exist_ok=True)
    train_path = os.path.join(temp_dir, f"train_dict_seed{seed}.txt")
    eval_path = os.path.join(temp_dir, f"eval_dict_seed{seed}.txt")
    
    # Write train dictionary
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_pairs))
    
    # Write eval dictionary
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_pairs))
    
    print(f"Created train dictionary with {len(train_pairs)} pairs at {train_path}")
    print(f"Created eval dictionary with {len(eval_pairs)} pairs at {eval_path}")
    
    return train_path, eval_path


def run_multiple_sampling_iterations(
    muse_config: MuseConfig,
    full_dict_path: str,
    train_size: int,
    iterations: int,
    base_seed: int
) -> Dict[str, float]:
    """
    Run multiple iterations of train/eval with different sampled dictionaries.
    
    Args:
        muse_config: Configuration for MUSE
        full_dict_path: Path to the complete dictionary file
        train_size: Number of pairs to include in the training dictionary
        iterations: Number of sampling iterations to run
        base_seed: Base random seed for reproducibility
        
    Returns:
        Dictionary of aggregated metrics across all iterations
    """
    all_metrics = {}
    
    print(f"Running {iterations} sampling iterations with train_size={train_size}")
    
    # Ensure paths are fully resolved before starting
    # muse_config = resolve_config_paths(muse_config)
    
    # # Verify that key paths are resolved and exist
    # for path_attr in ['src_emb', 'tgt_emb']:
    #     path = getattr(muse_config, path_attr)
    #     if not os.path.exists(path):
    #         print(f"[WARNING] Path does not exist: {path_attr}={path}")
    
    for i in range(iterations):
        seed = base_seed + i
        print(f"\n--- Iteration {i+1}/{iterations} (seed={seed}) ---")
        
        # Create sampled dictionaries
        train_path, eval_path = create_sampled_dictionaries(
            full_dict_path=full_dict_path,
            train_size=train_size,
            seed=seed
        )
        
        # Create a new config with the updated dictionary paths
        # Make sure all paths are fully resolved (not using ${...} syntax)
        iter_config = MuseConfig()
        for field in fields(muse_config):
            setattr(iter_config, field.name, getattr(muse_config, field.name))
        
        # Update the dictionary paths
        iter_config.dico_train = train_path
        iter_config.dico_eval = eval_path
        
        print(f"[DEBUG] iter_config.src_emb: {iter_config.src_emb}")
        print(f"[DEBUG] iter_config.tgt_emb: {iter_config.tgt_emb}")
        print(f"[DEBUG] iter_config.dico_train: {iter_config.dico_train}")
        print(f"[DEBUG] iter_config.dico_eval: {iter_config.dico_eval}")

        # Run supervised alignment
        result_metrics, topk_nn = muse_supervised(iter_config)
        
        print(f"Iteration {i+1} metrics: {result_metrics}")
        
        # Aggregate metrics
        for metric, value in result_metrics.items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)
    
    # Calculate mean and standard deviation
    aggregated_metrics = {}
    for metric, values in all_metrics.items():
        aggregated_metrics[f"{metric}_mean"] = np.mean(values)
        aggregated_metrics[f"{metric}_std"] = np.std(values)
    
    print(f"\n--- Aggregated Results ({iterations} iterations) ---")
    for metric, value in aggregated_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return aggregated_metrics


cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)

# TODO: parse MODEL_CONFIGS from yaml config
for model in MODEL_CONFIGS:
    cs.store(group="model", name=f"{model}", node=MODEL_CONFIGS[model])



@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the project"""
    setup_logging()
    
    # Resolve config and print
    OmegaConf.resolve(cfg)
    print(f"Run config:\n{'-' * 20}\n{OmegaConf.to_yaml(cfg)}{'-' * 20}\n")
    print("[DEBUG] Test print from alignment.py")


    print("[DEBUG] Reducing embeddings...")
    reduce_dimensions(cfg.muse)
    print("[DEBUG] Embeddings were reduced!")

    # Create a copy of the muse config, ensuring all paths are fully resolved
    muse_config = MuseConfig()
    for field in fields(muse_config):
        setattr(muse_config, field.name, getattr(cfg.muse, field.name))
        # if hasattr(cfg.muse, field.name):
        #     # Convert any OmegaConf values to Python native types
        #     value = OmegaConf.to_container(getattr(cfg.muse, field.name), resolve=True)
        #     setattr(muse_config, field.name, value)
    
    # Further resolve any remaining path variables (like ~, $HOME, etc.)
    # muse_config = resolve_config_paths(muse_config)

    # Check if we're using the sampling mode
    if hasattr(cfg.muse, 'use_sampling') and cfg.muse.use_sampling:
        full_dict_path = cfg.muse.full_dict_path
        train_size = cfg.muse.sample_train_size
        iterations = cfg.muse.sample_iterations
        
        # Ensure that all paths are fully resolved
        # muse_config = resolve_config_paths(muse_config)
        
        print(f"[DEBUG] Using sampling mode with {iterations} iterations")
        result_metrics = run_multiple_sampling_iterations(
            muse_config=muse_config,
            full_dict_path=full_dict_path,
            train_size=train_size,
            iterations=iterations,
            base_seed=cfg.common.seed
        )
        
        # Save results with sampling info in filename
        result_filename = f"{cfg.muse.lm}_{cfg.muse.vm}_dim{cfg.muse.emb_dim}_sampling_train{train_size}_iter{iterations}.json"
    else:
        # Regular single run
        if cfg.muse.supervised and cfg.muse.supervised_kfolds:
            result_metrics, topk_nn = muse_supervised_kfolds(muse_config)
            for i, fold_result in enumerate(result_metrics[:-1]):
                print()
                print(f"[DEBUG] FOLD {i + 1}: {fold_result}")
            print()
            print(f"[DEBUG] ALL DATA: {result_metrics[-1]}")
            
        elif cfg.muse.supervised:
            result_metrics, topk_nn = muse_supervised(muse_config)
            print(f"[DEBUG] result_metrics: {result_metrics}")
        else:
            raise NotImplemented("Unsupervised alignment is not supported yet.")
            
        result_filename = f"{cfg.muse.lm}_{cfg.muse.vm}_dim{cfg.muse.emb_dim}.json"
        topk_nn_filename = f"{cfg.muse.lm}_{cfg.muse.vm}_dim{cfg.muse.emb_dim}_top{cfg.muse.topk}_nn.json"

    print("[DEBUG] Embedding spaces were aligned.")

    with open(f"{cfg.muse.result_metrics_save_dir}/{result_filename}", 'w', encoding='utf-8') as f:
        json.dump(result_metrics, f, indent=4)

    if topk_nn != []:
        with open(f"{cfg.muse.result_metrics_save_dir}/topk/{topk_nn_filename}", 'w', encoding='utf-8') as f:
            json.dump(topk_nn, f, indent=4)

    print("[DEBUG] Metrics of alignment were saved.")




if __name__ == "__main__":
    load_dotenv("./.env")
    main() 