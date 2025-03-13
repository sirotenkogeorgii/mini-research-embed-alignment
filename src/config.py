from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch
import warnings

from omegaconf import II, MISSING


class ModelType(Enum):
    LM = "language_embeddings"
    VM = "vision_embeddings"


# class ComputationType(Enum):
#     FLOAT32 = "float32"
#     FLOAT16 = "float16"
#     BFLOAT16 = "bfloat16"


class ExperimentsType(Enum):
    IMAGE_DISP = "image_disp"
    LANG_DISP = "lang_disp"
    FREQ = "freq"
    POLY = "poly"
    BASE = "base"


class Language(Enum):
    ENGLISH = "english"
    GERMAN = "german"
    RUSSIAN = "russian"


@dataclass
class ModelInfo:
    model_type: ModelType = field(
        default=ModelType.LM, metadata={"help": "Model types: LM/VM"}
    )
    model_id: str = field(
        default="bert-base-uncased", metadata={"help": "Model to use."}
    )
    dim: int = field(default=768, metadata={"help": "size of dimension."})
    model_size: float = field(
        default=MISSING, metadata={"help": "Millions of Parameters in the model."}
    )
    model_name: str = field(init=False)
    # last_n_layers: int = field(
    #     default=1, metadata={"help": "Number of last layers to extract embeddings from."}
    # )
    last_n_hidden_states: Optional[int] = field(
        default=1, metadata={"help": "Number of last layers to extract embeddings from. Greater than 1."}
    )
    specific_last_hidden_state: Optional[int] = field(
        default=None, metadata={"help": "Hidden states from the last nth layer to extract. Greater than 1."}
    )
    use_quantization_8bit: bool = field(
        default=False, metadata={"help": "Whether to use model quantization."}
    )
    torch_type: str = field(
        default="float32", metadata={"help": "Mixed precision type."}
    )

    def __post_init__(self):
        if self.model_id in ["ViT-B/32", "ViT-L/14", "RN50", "RN101", "RN50x64"]:
            self.model_name = f"clip-{self.model_id.replace('/', '-')}"
        else:
            self.model_name = self.model_id.split("/")[-1]

        if self.last_n_hidden_states is not None and self.specific_last_hidden_state is not None:
            raise Exception("At least one of them must be None [last_n_hidden_states, specific_last_hidden_state]!")
        if self.last_n_hidden_states is None and self.specific_last_hidden_state is None:
            raise Exception("At least one of them must be non-None [last_n_hidden_states, specific_last_hidden_state]!")
    



MODEL_CONFIGS = {
    "Llama-2-7b": ModelInfo(
        model_id="meta-llama/Llama-2-7b-hf",
        model_size=6740,
        dim=4096,
        model_type=ModelType.LM,
    ),
    "opt-125m": ModelInfo(
        model_id="facebook/opt-125m",
        model_size=125,
        dim=768,
        model_type=ModelType.LM,
    ),
    "distilbert-base-uncased": ModelInfo(
        model_id="distilbert/distilbert-base-uncased",
        model_size=67,
        dim=768,
        model_type=ModelType.LM,
    ),
    "bert-base-uncased": ModelInfo(
        model_id="google-bert/bert-base-uncased",
        model_size=110,
        dim=768,
        model_type=ModelType.LM,
    ),
    "bert-base-multilingual-cased": ModelInfo(
        model_id="bert-base-multilingual-cased",
        model_size=110,
        dim=768,
        model_type=ModelType.LM,
    ),
    "clip-ViT-B-32": ModelInfo(
        model_id="ViT-B/32",
        model_size=151,
        dim=512,
        model_type=ModelType.VM,
    ),
    "clip-ViT-L-14": ModelInfo(
        model_id="ViT-L/14",
        model_size=427,
        dim=768,
        model_type=ModelType.VM,
    ),
}


@dataclass
class CommonConfig:
    seed: int = field(default=42, metadata={"help": "Seed for reproducibility."})
    embeddings_dataset_root: str = field(
        default="./data/embeddings",
        metadata={"help": "Path to save word embeddings (decontextualized)"},
    )
    emb_per_object: bool = field(
        default=False,
        metadata={"help": "Path to save one embedding per image or sentence"},
    )
    num_classes: int = field(
        default=1000000,
        metadata={"help": "Number of classes in the dataset."},
    )
    dictionary_path: str = field(
        default="./data/dicts",
        metadata={"help": "Path to save the dictionary."},
    )

    

@dataclass
class LanguageDataConfig:
    dataset_name_hf: str = field(
        default="jaagli/common-words-79k",
        metadata={"help": "Name of the dataset with aliases on Hugging Face."},
    )
    dataset_size: Optional[int] = field(
        default=None,
        metadata={"help": "How many aliases to extract."},
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for aliases."},
    )
    
@dataclass
class VisionDataConfig:
    dataset_name: str = field(
        default="imagenet", metadata={"help": "Name of the image dataset."}
    )
    dataset_path: str = field(
        default=MISSING,
        metadata={"help": "Path to image dataset."},
    )
    image_id_pairs: str = field(
        default=MISSING,
        metadata={"help": "Path to save the image id pairs."},
    )
    per_image: bool = field(
        default=False, metadata={"help": "Whether to save embeddings separately."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for aliases."},
    )


@dataclass
class DataConfig:
    vision_data: VisionDataConfig = field(default_factory=VisionDataConfig)
    language_data: LanguageDataConfig = field(default_factory=LanguageDataConfig)



@dataclass
class EndpointConfig:
    api_key: Optional[str] = field(
        default=None, metadata={"help": "API key for inference endpoints."}
    )
    lm_endpoint: Optional[str] = field(
        default=None, metadata={"help": "Endpoint URL for language model inference."}
    )
    vm_endpoint: Optional[str] = field(
        default=None, metadata={"help": "Endpoint URL for vision model inference."}
    )
    max_concurrent_requests: int = field(
        default=5, metadata={"help": "Maximum number of concurrent requests to HF endpoint."}
    )
    timeout: int = field(
        default=30, metadata={"help": "Timeout for API requests in seconds."}
    )


@dataclass
class MuseConfig:
    seed: int = field(
        default=II("common.seed"), metadata={"help": "Seed for reproducibility."}
    )
    exp_type: ExperimentsType = field(
        default=ExperimentsType.BASE,
        metadata={
            "help": "Different experiments settings: BASE, image_disp, lang_disp, freq, poly."
        },
    )
    lm: str = field(default="bert-base-uncased", metadata={"help": "Language model name."})
    vm: str = field(default="clip-ViT-B-32", metadata={"help": "Vision model name."})
    language: str = field(default="english", metadata={"help": "Language to use for alignment."})
    dim: int = field(default=768, metadata={"help": "Dimension of the embeddings."})
    fold: int = field(default=1, metadata={"help": "Fold number."})
    bin_name: str = field(
        default="",
        metadata={"help": "Various Bins name only for non-original experiments."},
    )
    data_type: str = field(default="cleaned", metadata={"help": "Data types: cleaned."})
    n_refinement: int = field(default=0, metadata={"help": "Number of refinements."})
    normalize_embeddings: str = field(
        default="center", metadata={"help": "Normalization."}
    )
    more_exp: bool = field(init=False)
    tgt_lang: str = field(init=False, metadata={"help": "Target language."})
    src_lang: str = field(init=False, metadata={"help": "Source language."})
    emb_dim: int = field(init=False, metadata={"help": "Dimension of the embeddings."})
    dico_eval: str = field(
        init=False, metadata={"help": "Path to load evaluation dictionary."}
    )
    dico_train: str = field(
        init=False, metadata={"help": "Path to load training dictionary."}
    )
    src_emb: str = field(
        init=False, metadata={"help": "Path to load source representations."}
    )
    tgt_emb: str = field(
        init=False, metadata={"help": "Path to load target representations."}
    )
    exp_name: str = field(
        default="", metadata={"help": "Path to log and store experiments."}
    )
    exp_path: str = field(
        default="", metadata={"help": "Path to log and store experiments."}
    )
    cuda: bool = field(default=True, metadata={"help": "Use GPU."})
    export: str = field(
        default="", metadata={"help": "Export embeddings after training (txt / pth)"}
    )

    # No need to change the following parameters
    exp_id: str = field(default="", metadata={"help": "Experiment ID."})
    max_vocab: int = field(
        default=200000, metadata={"help": "Maximum vocabulary size (-1 to disable)"}
    )
    dico_method: str = field(
        default="csls_knn_100",
        metadata={
            "help": "Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)"
        },
    )
    dico_build: str = field(
        default="S2T&T2S", metadata={"help": "S2T,T2S,S2T|T2S,S2T&T2S"}
    )
    dico_threshold: float = field(
        default=0, metadata={"help": "Threshold confidence for dictionary generation"}
    )
    dico_max_rank: int = field(
        default=10000, metadata={"help": "Maximum dictionary words rank (0 to disable)"}
    )
    dico_min_size: int = field(
        default=0, metadata={"help": "Minimum dictionary size (0 to disable)"}
    )
    dico_max_size: int = field(
        default=0, metadata={"help": "Maximum dictionary size (0 to disable)"}
    )
    verbose: int = field(default=2, metadata={"help": "Verbosity level."})
    load_optim: bool = field(
        default=False, metadata={"help": "Load optimized results."}
    )
    dico_root: str = field(
        default=f"{II('common.dictionary_path')}/{II('dataset.dataset_name')}",
        metadata={"help": "Path to save the dictionary."},
    )
    vm_emb_root: str = field(
        default=f"{II('common.embeddings_dataset_root')}/{ModelType.VM.value}/{II('dataset.dataset_name')}",
        metadata={"help": "Path to save the vision model embeddings."},
    )
    lm_emb_root: str = field(
        default=f"{II('common.embeddings_dataset_root')}/{ModelType.LM.value}",
        metadata={"help": "Path to save the language model embeddings."},
    )

    def __post_init__(self):
        self.more_exp = True if self.exp_type != ExperimentsType.BASE else False
        exp_dict_folders = {
            ExperimentsType.IMAGE_DISP: f"{self.vm}_disp",
            ExperimentsType.LANG_DISP: f"{self.vm}_disp",
            ExperimentsType.POLY: "poly",
            ExperimentsType.FREQ: "freq",
            ExperimentsType.BASE: "base",
        }

        test_dict_folder = exp_dict_folders.get(self.exp_type, "base")
        self.src_lang = self.vm
        self.tgt_lang = self.lm
        self.emb_dim = self.dim
        self.dico_train = f"{self.dico_root}/{self.exp_type.value}/{self.language}/train_{self.fold}_{self.data_type}.txt"
        self.dico_eval = (
            f"{self.dico_root}/{test_dict_folder}/{self.language}/test_{self.fold}_{self.data_type}.txt"
        )
        self.src_emb = f"{self.vm_emb_root}/{self.vm}_{self.emb_dim}.pth"
        self.tgt_emb = f"{self.lm_emb_root}/{self.language}/{self.lm}_{self.emb_dim}.pth"


@dataclass
class HuggingFaceRepository:
    token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging face access token."}
    )
    repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository name."}
    )
    repo_type: str = field(
        default="dataset",
        metadata={"help": "Type of the repository."}
    )
    intermediate_buffer_save: bool = field(
        default=True, 
        metadata={"help": "Before uploading to Hugging Face save it in buffer."})
    

@dataclass
class RunConfig:
    model: ModelInfo = field(default_factory=lambda: MODEL_CONFIGS["bert-base-uncased"])
    common: CommonConfig = field(default_factory=CommonConfig)
    dataset: DataConfig = field(default_factory=DataConfig)
    muse: MuseConfig = field(default_factory=MuseConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    run_muse: bool = field(default=False, metadata={"help": "Run MUSE part."})
    
    save_hugging_face: bool = field(default=False, metadata={"help": "Save files on hugging face."})
    hugging_face_save_repo: HuggingFaceRepository = field(default_factory=HuggingFaceRepository)
