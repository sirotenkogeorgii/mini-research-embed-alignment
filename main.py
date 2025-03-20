import hydra
import logging
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.config import MODEL_CONFIGS, RunConfig
from src.procrustes import MuseExp
from src.rep_extractor import RepExtractor
from src.utils.translation_utils import Translator, create_dummy_dataset
# from src.utils.utils_helper import reduce_dim


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_reps(args: DictConfig) -> None:
    """Extract embeddings for the model specified in the config"""
    rep_extractor = RepExtractor(config=args)
    rep_extractor.process_embeddings()
    print("-" * 25 + "Extract and Decontextualize representation completed!" + "-" * 25)

    # # Optionally reduce dimensions 
    # if args.get("reduce_dim", False):
    #     reduce_dim(args, MODEL_CONFIGS)


def run_muse(args: DictConfig) -> None:
    """Run MUSE alignment experiments"""
    procrustes_exp = MuseExp(args)
    procrustes_exp.run(data_type=args.muse.data_type, exp_type=args.muse.exp_type)


def process_all_languages(args: DictConfig) -> None:
    """Process embeddings for all languages"""
    original_language = args.muse.language
    
    for language in args.language.enabled_languages:
        logging.info(f"Processing language: {language}")
        # Update config for current language
        args.muse.language = language
        
        # Run get_reps with language-specific configuration
        get_reps(args)
        
        # Optionally run MUSE
        if args.run_muse:
            run_muse(args)
    
    # Restore original language setting
    args.muse.language = original_language


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
    print("[DEBUG] Test print from main.py")

    # Prepare translations if needed
    # if cfg.get("prepare_translations", False):
    #     prepare_translations(cfg)
    
    # If processing a specific language or model
    if not cfg.get("process_all_languages", False):
        # Process embeddings for the specified language and model
        get_reps(cfg)
        
        # Run MUSE if specified
        if cfg.run_muse:
            run_muse(cfg)
    else:
        # Process all languages
        process_all_languages(cfg)


if __name__ == "__main__":
    load_dotenv("./.env")
    main() 