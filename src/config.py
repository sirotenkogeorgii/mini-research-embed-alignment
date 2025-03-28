from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any

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
        self.model_name = self.model_id.split("/")[-1]

        if self.last_n_hidden_states is not None and self.specific_last_hidden_state is not None:
            raise Exception("At least one of them must be None [last_n_hidden_states, specific_last_hidden_state]!")
        if self.last_n_hidden_states is None and self.specific_last_hidden_state is None:
            raise Exception("At least one of them must be non-None [last_n_hidden_states, specific_last_hidden_state]!")
    

# 128, 256, 512, 768, 1024, 1280, 4096, 

MODEL_CONFIGS = {
    "llama-2-7b": ModelInfo(
        model_id="meta-llama/Llama-2-7b-hf",
        model_size=6740,
        dim=4096,
        model_type=ModelType.LM,
    ),
    "distilbert-base-uncased": ModelInfo(
        model_id="distilbert/distilbert-base-uncased",
        model_size=67,
        dim=768,
        model_type=ModelType.LM,
    ),
    "bert-medium": ModelInfo(
        model_id="prajjwal1/bert-medium",
        model_size=41,
        dim=512,
        model_type=ModelType.LM,
    ),
    "bert-base-uncased": ModelInfo(
        model_id="google-bert/bert-base-uncased",
        model_size=110,
        dim=768,
        model_type=ModelType.LM,
    ),

    ####### russian langue models #######
    "bert-base-cased-ru": ModelInfo(
        model_id="DeepPavlov/rubert-base-cased",
        model_size=110,
        dim=768,
        model_type=ModelType.LM,
    ),
    "ruBert-base": ModelInfo(
        model_id="ai-forever/ruBert-base",
        model_size=178,
        dim=768,
        model_type=ModelType.LM,
    ),
    "ruBert-large": ModelInfo(
        model_id="ai-forever/ruBert-large",
        model_size=427,
        dim=1024,
        model_type=ModelType.LM,
    ),
    #####################################

    "bert-large-uncased": ModelInfo(
        model_id="google-bert/bert-large-uncased",
        model_size=340,
        dim=1024,
        model_type=ModelType.LM,
    ),

    "opt-125m": ModelInfo(
        model_id="facebook/opt-125m",
        model_size=125,
        dim=768,
        model_type=ModelType.LM,
    ),
    "opt-350m": ModelInfo(
        model_id="facebook/opt-350m",
        model_size=350,
        dim=1024,
        model_type=ModelType.LM,
    ),
    "opt-1.3b": ModelInfo(
        model_id="facebook/opt-1.3b",
        model_size=1300,
        dim=2048,
        model_type=ModelType.LM,
    ),

    "gpt2": ModelInfo(
        model_id="openai-community/gpt2",
        model_size=117,
        dim=768,
        model_type=ModelType.LM,
    ),
    "gpt2-medium": ModelInfo(
        model_id="openai-community/gpt2-medium",
        model_size=345,
        dim=1024,
        model_type=ModelType.LM,
    ),
    "gpt2-large": ModelInfo(
        model_id="openai-community/gpt2-large",
        model_size=774,
        dim=1280,
        model_type=ModelType.LM,
    ),

    "vit-base-patch16-224": ModelInfo(
        model_id="google/vit-base-patch16-224",
        model_size=86,
        dim=768,
        model_type=ModelType.VM,
    ),
    "vit-large-patch16-224": ModelInfo(
        model_id="google/vit-large-patch16-224",
        model_size=304,
        dim=1024,
        model_type=ModelType.VM,
    ),
    "dino-vits16": ModelInfo(
        model_id="facebook/dino-vits16",
        model_size=22,
        dim=384,
        model_type=ModelType.VM,
    ),
    "dino-vitb16": ModelInfo(
        model_id="facebook/dino-vitb16",
        model_size=86,
        dim=768,
        model_type=ModelType.VM,
    ),
    "resnet18": ModelInfo(
        model_id="resnet18",
        dim=512,
        model_size=512,
        model_type=ModelType.VM,
    ),
    "resnet34": ModelInfo(
        model_id="resnet34",
        dim=512,
        model_type=ModelType.VM,
    ),
    "resnet50": ModelInfo(
        model_id="resnet50",
        dim=2048,
        model_type=ModelType.VM,
    ),
    "resnet101": ModelInfo(
        model_id="resnet101",
        dim=2048,
        model_type=ModelType.VM,
    ),
    "resnet152": ModelInfo(
        model_id="resnet152",
        dim=2048,
        model_type=ModelType.VM,
    ),
    "efficientnet_b0": ModelInfo(
        model_id="efficientnet_b0",
        dim=1280,
        model_type=ModelType.VM,
    ),
    "efficientnet_b2": ModelInfo(
        model_id="efficientnet_b2",
        dim=1408,
        model_type=ModelType.VM,
    ),
    "efficientnet_b4": ModelInfo(
        model_id="efficientnet_b4",
        dim=1792,
        model_type=ModelType.VM,
    ),
    "efficientnet_b6": ModelInfo(
        model_id="efficientnet_b6",
        dim=2304,
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
    # reduce_dim: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to reduce the dimencionality of embeddings."},
    # )
    # reductiom_dims: List[int] = field(default_factory=lambda: [])

    

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
    available_image_ids: str = field(
        default=MISSING,
        metadata={"help": "Available image ids."},
    )
    per_image: bool = field(
        default=False, metadata={"help": "Whether to save embeddings separately."}
    )
    max_per_class_images: int = field(
        default=100, metadata={"help": "Maximum images per class to use for a concept extraction."}
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
    supervised: bool = field(default=True, metadata={"help": "Whether to perform alignment in a supervised manner."})
    lm_dataset: str = field(default="common-words-79k", metadata={"help": "Source dataset for the language embeddings."})
    vm_dataset: str = field(default="imagenet-ul-ex-1k-train-subset", metadata={"help": "Source dataset for the vision embeddings."})
    lm: str = field(default="bert-base-uncased", metadata={"help": "Language model name."})
    vm: str = field(default="resnet18", metadata={"help": "Vision model name."})
    language: str = field(default="english", metadata={"help": "Language to use for alignment."})
    dim: int = field(default=768, metadata={"help": "Dimension of the embeddings."})
    layer_identifier: str = field(default="layer_1", metadata={"help": "From what layer the language embeddings were extracted"})
    
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

    src_dico: Any = field(
        default=None, metadata={"help": "Path to load source representations."}
    )
    tgt_dico: Any = field(
        default=None, metadata={"help": "Path to load target representations."}
    )

    src_emb: str = field(
        default="", metadata={"help": "Path to load source representations."}
    )
    tgt_emb: str = field(
        default="", metadata={"help": "Path to load target representations."}
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
        default=1000, metadata={"help": "Maximum dictionary size (0 to disable)"}
    )
    verbose: int = field(default=2, metadata={"help": "Verbosity level."})
    load_optim: bool = field(
        default=False, metadata={"help": "Load optimized results."}
    )

    result_metrics_save_dir: str = field(
        default=f"",
        metadata={"help": "Path to save the resulting metrics of alignment."}
    )

    dico_train: str = field(
        default=MISSING,
        metadata={"help": "Path to dico train."},
    )

    dico_eval: str = field(
        default=MISSING,
        metadata={"help": "Path to dico eval."},
    )

    full_dict_path: str = field(
        default="",
        metadata={"help": "Path to full dico."},
    )
    use_sampling: bool = field(default=False)
    sample_train_size: int = field(default=1000)
    sample_iterations: int = field(default=5)

    topk: int = field(default=100)
    save_embeddings_dir: str = field(default="")
    save_word2id_dir: str = field(default="")
    # save_word2id_dir: str = field(default="")

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
        self.exp_name = "test_alignment"
        self.src_lang = self.vm
        self.tgt_lang = self.lm
        self.emb_dim = min(MODEL_CONFIGS[self.lm].dim, MODEL_CONFIGS[self.vm].dim)
        self.src_emb = f"{II('common.embeddings_dataset_root')}/{ModelType.VM.value}/{self.vm_dataset}/aggregated/{self.vm}_{self.emb_dim}.pth"
        self.tgt_emb = f"{II('common.embeddings_dataset_root')}/{ModelType.LM.value}/{self.language}/{self.layer_identifier}/{self.lm_dataset}/{self.lm}_{self.emb_dim}.pth"
    


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
    
    save_hugging_face: bool = field(default=False, metadata={"help": "Whether to save files on hugging face."})
    hugging_face_save_repo: HuggingFaceRepository = field(default_factory=HuggingFaceRepository)
