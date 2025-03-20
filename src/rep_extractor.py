import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import ModelType
from src.utils.llm_rep_utils import LMEmbedding
from src.utils.vm_rep_utils import VMEmbedding
from src.utils.file_saver import FileSaver


class RepExtractor:
    def __init__(self, config: DictConfig) -> None:
        self.config: DictConfig = config

        # self.model_name: str = config.model.model_name
        self.model_type: ModelType = config.model.model_type
        # self.model_dim: int = config.model.dim
        # self.seed: int = config.common.seed
        # self.current_language: str = config.muse.language
        
        self.file_saver = FileSaver(self.config)

        self.config.common.embeddings_dataset_root = f"{config.common.embeddings_dataset_root}/{self.model_type.value}"
        # self.config.common.embeddings_dataset_root = f"{config.common.embeddings_dataset_root}/{config.model.model_type.value}"
        
        
    def process_embeddings(self) -> None:
        if self.model_type == ModelType.VM:
            self.__process_vision_embeddings()
            return
        elif self.model_type == ModelType.LM:
            self.__process_language_embeddings()

    def __process_vision_embeddings(self) -> None:
        """Process vision embeddings for all languages - same images for all languages"""
        # with open(self.config.dataset.vision_data.image_id_pairs, "r") as f:
        with open(self.config.dataset.vision_data.available_image_ids, "r") as file:
            # image id pairs are "n01983481": ["american_lobster", "maine_lobster"]
            # image_id_pairs = json.load(f)
            # image_ids = [i for i in list(image_id_pairs.keys())] # n04357314 like keys
            lines = file.readlines()
            image_ids = [line.strip() for line in lines]

        self.__get_vm_rep(image_ids)

    def __process_language_embeddings(self) -> None:
        """Process language embeddings for the current language"""
        self.__get_lm_rep()

    def __get_vm_rep(self, image_labels: list) -> None:
        embeddings_extractor = VMEmbedding(self.config, image_labels, self.file_saver)
        embeddings_extractor.get_vm_layer_representations()

    def __get_lm_rep(self) -> None:
        embeddings_extractor = LMEmbedding(self.config, self.file_saver)
        embeddings_extractor.get_lm_layer_representations() 