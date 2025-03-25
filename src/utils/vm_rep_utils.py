import json
import logging
import os
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional, Union

# import clip
import numpy as np
import requests
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from img2vec_pytorch import Img2Vec
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoFeatureExtractor


class ImageDataset(Dataset):
    def __init__(self, dataset_path, image_classes, extractor, resolution=224) -> None:
        super(ImageDataset, self).__init__()
        self.dataset_path: Path = dataset_path
        self.labels: list = image_classes
        self.extractor: Any = extractor
        self.MAX_SIZE: int = 200 # maximum possible images per label
        self.RESOLUTION_HEIGHT: int = resolution
        self.RESOLUTION_WIDTH: int = resolution
        self.CHANNELS: int = 3

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Tuple[str, int]]:
        images = []
        # print("[DEBUG](ImageDataset)(0)")
        category_path = self.dataset_path / self.labels[index]
        for filename in category_path.iterdir():
            try:
                images.append(Image.open(category_path / filename).convert("RGB"))
            except:
                print("Failed to pil", filename)

        # print("[DEBUG](ImageDataset)(1)")

        category_size = len(images)
        inputs = torch.zeros(
            self.MAX_SIZE, self.CHANNELS, self.RESOLUTION_HEIGHT, self.RESOLUTION_WIDTH
        )
        # print("[DEBUG](ImageDataset)(2)")
        try:
            values = self.extractor(images=images, return_tensors="pt")
            with torch.no_grad():
                inputs[:category_size, :, :, :].copy_(values.pixel_values) # TODO: copy (e)
        except:
            print("*" * 20 + "Failed to extract" + "*" * 20, category_path)
        # print("[DEBUG](ImageDataset)(3)")

        return inputs, (self.labels[index], category_size)


class VMEmbedding:
    def __init__(self, config: DictConfig, image_labels: List[str], file_saver):
        self.config: DictConfig = config
        self.model_id: str = config.model.model_id
        self.bs = config.dataset.vision_data.batch_size
        self.model_name: str = config.model.model_name
        self.model_dim = config.model.dim
        self.dataset_name: str = config.dataset.vision_data.dataset_name
        self.dataset_path: str = Path(config.dataset.vision_data.dataset_path)
        self.labels: List[str] = image_labels
        self.max_per_class_images = config.dataset.vision_data.max_per_class_images
        self.save_embeddings_path: Path = (
                # Path(config.common.embeddings_dataset_root) / self.current_language / self.dataset_name
                Path(config.common.embeddings_dataset_root) / self.dataset_name
        )
        self.file_saver = file_saver
        self.per_image = config.dataset.vision_data.per_image
        
        # Create directory
        # os.makedirs(self.alias_emb_dir, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_vm_layer_representations(self) -> None:
        """Extract vision model representations"""

        file_to_save = self.save_embeddings_path / f"{self.model_name}_{self.model_dim}.pth"
        if file_to_save.exists():
            print(f"[DEBUG] File {file_to_save} already exists.")
            return

        if self.config.endpoint.lm_endpoint is not None:
            raise NotImplementedError("Endpoint support is not implemented yet.")
        else:
            self._process_with_local_model()
            
        print(f"Vision embeddings successfully extracted for {self.dataset_name}!")



    def _process_with_local_model(self) -> None:
        resolution = 224
        if self.model_name.startswith(("res", "efficient")):
            # categories_encode - list of tensors, image_categories - list of strings
            categories_encode, image_categories = self.__get_img2vec_pytorch_reps()
        else:
            categories_encode, image_categories = self.__get_hugging_face_reps(resolution)
        # elif self.model_name.startswith("seg") or self.model_name.startswith("vit"):
        #     categories_encode, image_categories = self.__get_hugging_face_reps(resolution)
        # else:
        #     categories_encode, image_categories = self.__get_clip_reps()


        # print(f"[DEBUG] categories_encode[0].shape: {categories_encode[0].shape}")
        embeddings = np.vstack(categories_encode)
        # print(f"[DEBUG] embeddings.shape: {embeddings.shape}")
        embeddinds_dim = embeddings.shape[1]
        # embegginds_dim = embeddings.shape[-1]
        save_path = self.save_embeddings_path / "aggregated" / f"{self.model_name}_{embeddinds_dim}.pth"
        self.save_embeddings(embeddings, image_categories, save_path)



    def __save_per_object_reps(
            self, num_images: int, reps: np.ndarray, label: str
        ) -> None:
            images_name = [f"{label}_{i}" for i in range(num_images)]
            embeddinds_dim = reps.shape[1]
            save_per_path = self.save_embeddings_path / "separated" / f"{self.model_name}_{embeddinds_dim}" / f"{label}.pth"

            # if not self.config.save_hugging_face:
            #     if not save_per_path.exists():
            #         save_per_path.mkdir(parents=True, exist_ok=True)

            self.save_embeddings(reps, images_name, save_per_path)



    def __get_img2vec_pytorch_reps(self) -> Tuple[list, list]:
        img2vec = Img2Vec(model=self.model_name, cuda=torch.cuda.is_available())
        use_tensor = not self.model_name.startswith("efficient")
        categories_encode = []
        image_categories = []
        for idx in range(len(self.labels)):
            # print(f"[DEBUG] ({idx + 1}) Processing label {self.labels[idx]}...")
            images = []
            category_path = self.dataset_path / self.labels[idx]
            if not category_path.exists(): 
                # print(f"[DEBUG] Image class {self.labels[idx]} was not found.")
                continue
            for file_i, filename in enumerate(category_path.iterdir()):
                try:
                    images.append(Image.open(category_path / filename).convert("RGB"))
                    if file_i + 1 == self.max_per_class_images:
                        break
                except:
                    print("Failed to PIL:", filename)
            with torch.no_grad():
                # print(f"[DEBUG] Images for a concept extraction: {len(images)}.")
                reps = img2vec.get_vec(images, tensor=use_tensor)
                if use_tensor:
                    reps = reps.to("cpu").numpy().squeeze()
                # print(f'[DEBUG] reps.shape: {reps.shape}')
                # print(f'[DEBUG] reps.mean(axis=0).shape: {reps.mean(axis=0).shape}')
                categories_encode.append(reps.mean(axis=0))
                image_categories.append(self.labels[idx])

                if self.per_image:
                    self.__save_per_object_reps(
                        num_images=len(images),
                        reps=reps,
                        label=self.labels[idx],
                    )
            print()

        return categories_encode, image_categories



    def __get_hugging_face_reps(self, resolution) -> Tuple[list, list]:
        cache_path = (
            Path.home() / ".cache/huggingface/transformers/models" / self.model_id
        )
        #  handles all preprocessing required to convert raw images into the model's expected input format
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, cache_dir=cache_path
        )
        # print("[DEBUGGGGGGGG](-2) Hi!")
        if not self.model_name.startswith("resnet"):
            # actual model
            model = AutoModel.from_pretrained(
                self.model_id,
                cache_dir=cache_path,
                output_hidden_states=True,
                return_dict=True,
            )
            model = model.to(self.device)

        model = model.eval()

        # print("[DEBUGGGGGGGG](-1) Hi!")
        imageset = ImageDataset(
            # self.image_dir, self.labels, feature_extractor, resolution
            self.dataset_path, self.labels, feature_extractor, resolution
        )
        # print("[DEBUGGGGGGGG](0) Hi!")
        image_dataloader = torch.utils.data.DataLoader(
            imageset, batch_size=self.bs, num_workers=4, pin_memory=True
        )
        # print("[DEBUGGGGGGGG](1) Hi!")

        # images_name = []
        categories_encode = []
        image_categories = []

        # iterate over labels in abtch (__getitem__ returns all preprocessed images for labels in batch)
        for inputs, (names, category_size) in tqdm(image_dataloader):
            # print("[DEBUGGGGGGGG](2) Hi!")
            inputs_shape = inputs.shape
            inputs = inputs.reshape(
                -1, inputs_shape[2], inputs_shape[3], inputs_shape[4]
            ).to(self.device)

            with torch.no_grad():
                # print("[DEBUGGGGGGGG](3) Hi!")
                outputs = model(pixel_values=inputs)
                if self.model_name.startswith("vit"):
                    chunks = torch.chunk(
                        # outputs.hidden_states[-1][:, 1:, :].cpu(),
                        outputs.hidden_states[-2][:, 1:, :].cpu(), # in ViT we test to extract last hidden state
                        inputs_shape[0], # number of batches
                        dim=0,
                    )
                elif self.model_name.startswith("dino"):
                    chunks = torch.chunk(
                        outputs.hidden_states[-1][:, 1:, :].cpu(),
                        inputs_shape[0], # number of batches
                        dim=0,
                    )
                else:
                    chunks = torch.chunk(
                        outputs.last_hidden_state.cpu(), 
                        inputs_shape[0], # number of batches
                        dim=0
                    )

                # features for every image (iterate over images)
                for idx, chip in enumerate(chunks):
                    # if self.model_name.startswith("vit"):
                    if self.model_name.startswith(("dino", "vit")):
                        images_features = np.mean(
                            chip[:category_size[idx]].numpy(),
                            axis=1,
                            keepdims=True,
                        ).squeeze()
                    else:
                        images_features = np.mean(
                            chip[:category_size[idx]].numpy(),
                            axis=(2, 3),
                            keepdims=True,
                        ).squeeze()
                    # features for categories
                    category_feature = np.expand_dims(images_features.mean(axis=0), 0)
                    image_categories.append(names[idx])
                    categories_encode.append(category_feature)

                    if self.per_image:
                        self.__save_per_object_reps(
                            num_images=category_size[idx],
                            reps=images_features,
                            label=names[idx],
                        )

        return categories_encode, image_categories

    

    def save_embeddings(self, vision_embeddings, image_categories, save_path) -> None:
        object_to_save = {"dico": image_categories, "vectors": torch.from_numpy(vision_embeddings).float()}
        self.file_saver.save(object_to_save, save_path)