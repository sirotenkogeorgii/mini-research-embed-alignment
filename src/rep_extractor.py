import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import ModelType
from src.utils.llm_rep_utils import LMEmbedding
from src.utils.vm_rep_utils import VMEmbedding


class RepExtractor:
    def __init__(self, config: DictConfig) -> None:
        self.config: DictConfig = config
        self.dataset_name: str = config.dataset.dataset_name


        # TODO: Uncomment and fix when I will do vision part
        # self.image_id_pairs: str = config.dataset.image_id_pairs


        self.model_name: str = config.model.model_name
        self.model_type: ModelType = config.model.model_type
        self.model_dim: int = config.model.dim
        self.alias_emb_dir: Path = (
                Path(config.common.alias_emb_dir) / self.model_type.value
        )
        self.seed: int = config.common.seed
        self.current_language: str = config.muse.language
        
        # Layer selection configuration
        
        # Embedding storage options
        self.emb_per_object: bool = config.common.emb_per_object

    def process_embeddings(self) -> None:
        if self.model_type == ModelType.VM:
        # TODO: Uncomment and fix when I will do vision part
            # self.__process_vision_embeddings()
            return
        elif self.model_type == ModelType.LM:
            self.__process_language_embeddings()

# TODO: Uncomment and fix when I will do vision part
    # def __process_vision_embeddings(self) -> None:
    #     """Process vision embeddings for all languages - same images for all languages"""
    #     self.alias_emb_dir = self.alias_emb_dir / self.dataset_name
    #     if not self.alias_emb_dir.exists():
    #         self.alias_emb_dir.mkdir(parents=True, exist_ok=True)

    #     with open(self.image_id_pairs, "r") as f:
    #         image_id_pairs = json.load(f)
    #         image_ids = [i for i in list(image_id_pairs.keys())]

    #     save_file_path = self.alias_emb_dir / f"{self.model_name}_{self.model_dim}.pth"
    #     if save_file_path.exists():
    #         print(f"File {save_file_path} already exists.")
    #         return
            
    #     self.__get_vm_rep(image_ids)

    def __process_language_embeddings(self) -> None:
        """Process language embeddings for the current language"""
        # Create language-specific directory
        # lang_dir = self.alias_emb_dir / self.current_language
        # if not lang_dir.exists():
        #     lang_dir.mkdir(parents=True, exist_ok=True)

        # save_file_path = lang_dir / f"{self.model_name}_{self.model_dim}.pth"
        # # TODO: add an option to override if exists
        # if save_file_path.exists():
        #     print(f"File {save_file_path} already exists.")
        #     return
            
        self.__get_lm_rep()

    def __get_vm_rep(self, image_labels: list) -> None:
        # TODO: rename to ...Extractor instead of ...Embedding.
        embeddings_extractor = VMEmbedding(self.config, image_labels)
        embeddings_extractor.get_vm_layer_representations()

    def __get_lm_rep(self) -> None:
        embeddings_extractor = LMEmbedding(self.config)
        embeddings_extractor.get_lm_layer_representations() 