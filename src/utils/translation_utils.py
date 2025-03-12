import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
from omegaconf import DictConfig
from tqdm import tqdm


class Translator:
    def __init__(self, config: DictConfig):
        self.config = config
        self.dataset_name = config.dataset.dataset_name
        self.languages = config.language.enabled_languages
        self.hf_api_key = config.hf_endpoint.api_key
        self.translation_dir = Path("data/translations")
        
        # Create translation directories
        for language in self.languages:
            lang_dir = self.translation_dir / language
            os.makedirs(lang_dir, exist_ok=True)
            
    def translate_dataset(self, source_file: str) -> None:
        """Translate datasets for all enabled languages"""
        if not Path(source_file).exists():
            logging.error(f"Source file {source_file} does not exist")
            return
            
        # Load source dataset (English)
        with open(source_file, 'r') as f:
            dataset = json.load(f)
            
        # For each language, translate the dataset
        for language in self.languages:
            # Skip English (source language)
            if language.lower() == "english":
                # Just save the original dataset
                output_file = self.translation_dir / language / f"{self.dataset_name}_translations.json"
                with open(output_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
                print(f"Saved original English dataset to {output_file}")
                continue
                
            # Check if translation already exists
            output_file = self.translation_dir / language / f"{self.dataset_name}_translations.json"
            if output_file.exists():
                print(f"Translation for {language} already exists: {output_file}")
                continue
                
            # Translate dataset
            translated_dataset = self._translate_dataset_to_language(dataset, language)
            
            # Save translated dataset
            with open(output_file, 'w') as f:
                json.dump(translated_dataset, f, indent=2)
                
            print(f"Saved translated dataset for {language} to {output_file}")
    
    def _translate_dataset_to_language(self, dataset: Dict, target_language: str) -> Dict:
        """Translate the dataset to the specified language"""
        translated_dataset = {
            "concept": dataset["concept"].copy(),  # Keep concept names in English
            "sentences": []
        }
        
        # Process each concept's sentences
        for concept_idx, sentences in enumerate(dataset["sentences"]):
            translated_sentences = []
            concept = dataset["concept"][concept_idx]
            
            print(f"Translating concept {concept_idx+1}/{len(dataset['concept'])}: {concept} to {target_language}")
            
            # Translate each sentence
            for sentence in tqdm(sentences):
                translated_sentence = self._translate_text(sentence, target_language)
                translated_sentences.append(translated_sentence)
                
            translated_dataset["sentences"].append(translated_sentences)
            
        return translated_dataset
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate a single text using HF Inference API or a fallback method"""
        if self.config.hf_endpoint.api_key:
            # Use HF Inference API for translation
            return self._translate_with_hf_api(text, target_language)
        else:
            # Fallback translation (mocked for demonstration)
            return self._mock_translation(text, target_language)
    
    def _translate_with_hf_api(self, text: str, target_language: str) -> str:
        """Translate text using HuggingFace Inference API"""
        # Map language to language code
        language_codes = {
            "english": "en",
            "german": "de",
            "russian": "ru"
        }
        
        target_code = language_codes.get(target_language.lower(), "en")
        
        # HF Translation endpoint
        # Note: In a real implementation, you would use a specific translation model endpoint
        model_url = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
        
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text,
            "parameters": {
                "src_lang": "en_XX",
                "tgt_lang": f"{target_code}_XX"
            }
        }
        
        try:
            response = requests.post(
                model_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("translation_text", text)
            else:
                logging.warning(f"Unexpected translation response format: {result}")
                return text
                
        except Exception as e:
            logging.error(f"Translation API error: {str(e)}")
            return text
    
    def _mock_translation(self, text: str, target_language: str) -> str:
        """Mock translation function for demonstration purposes"""
        # In a real implementation, this would use a local translation library
        # or a different API service
        
        prefixes = {
            "german": "[DE] ",
            "russian": "[RU] "
        }
        
        prefix = prefixes.get(target_language.lower(), "")
        return f"{prefix}{text}"  # Just prefix the original text for demonstration


def create_dummy_dataset(output_file: str, num_concepts: int = 10, sentences_per_concept: int = 5) -> None:
    """Create a dummy dataset for demonstration purposes"""
    concepts = [f"concept_{i}" for i in range(num_concepts)]
    
    dataset = {
        "concept": concepts,
        "sentences": []
    }
    
    for concept in concepts:
        sentences = [f"This is a sentence about {concept} number {j}." for j in range(sentences_per_concept)]
        dataset["sentences"].append(sentences)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Created dummy dataset with {num_concepts} concepts at {output_file}") 