import json
from pathlib import Path

import io
import torch
from omegaconf import DictConfig

from huggingface_hub import HfApi


class FileSaver:
    def __init__(self, config: DictConfig) -> None:
        self.config: DictConfig = config

        if self.config.save_hugging_face:
            self.hugging_face_api = HfApi(token=self.config.hugging_face_save_repo.token)
            self.hugging_face_repo = self.config.hugging_face_save_repo.repo_id
            self.hugging_face_repo_type = self.config.hugging_face_save_repo.repo_type

    def save(self, file_object, save_path):
        # save_path must contain the name of the file as well

        if self.config.save_hugging_face:
            if isinstance(save_path, Path): save_path = str(save_path)

            object_to_save = file_object
            if self.config.hugging_face_save_repo.intermediate_buffer_save:
                object_to_save = io.BytesIO()
                torch.save(file_object, object_to_save)  # Save directly to buffer (no file)
                object_to_save.seek(0)

            self.hugging_face_api.upload_file(
                path_or_fileobj=object_to_save,
                path_in_repo=save_path, # starts with the root ("filename")
                repo_id=self.hugging_face_repo,
                repo_type=self.hugging_face_repo_type
            )
        
        else:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(file_object, save_path)