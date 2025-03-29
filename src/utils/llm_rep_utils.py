import re
import json
import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional, Union

import string
import pymorphy2

import duckdb
import numpy as np
import torch
import requests
import warnings
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LMEmbedding:
    def __init__(self, args: DictConfig, file_saver):
        self.config: DictConfig = args
        self.model_id: str = args.model.model_id
        self.bs = self.config.dataset.language_data.batch_size
        self.model_name: str = args.model.model_name
        self.model_dim = args.model.dim
        self.dataset_name: str = args.dataset.language_data.dataset_name_hf.split('/')[-1]
        self.current_language: str = args.muse.language
        self.file_saver = file_saver

        if self.config.model.specific_last_hidden_state is not None:
            layers_to_extract_str = f"layer_{self.config.model.specific_last_hidden_state}"
        else:
            layers_to_extract_str = f"layers_1-{self.config.model.last_n_hidden_states}"
        self.save_embeddings_path: Path = (
                Path(args.common.embeddings_dataset_root) / self.current_language / layers_to_extract_str / self.dataset_name
        )
        

        self.morph = pymorphy2.MorphAnalyzer()
        
        # Store embedding options
        self.emb_per_object: bool = args.common.emb_per_object
        
        # Layer selection options
        self.last_n_hidden_states: int = args.model.last_n_hidden_states
        
        self.device: list = (
            [i for i in range(torch.cuda.device_count())]
            if torch.cuda.device_count() >= 1
            else ["cpu"]
        )
        
        # Create directory structure
        os.makedirs(self.save_embeddings_path, exist_ok=True)
        
        # Initialize database
        self.con = duckdb.connect(f"{self.save_embeddings_path}/{self.model_name}.db")
        self.con.execute("DROP TABLE IF EXISTS data")
        self.con.commit()
        self.con.execute(f"CREATE TABLE IF NOT EXISTS data (alias VARCHAR, embedding FLOAT[{self.model_dim}])")

        self.skipped_sentences = 0
        self.processed_sentences = 0

    def get_lm_layer_representations(self) -> None:
        """Extract language model representations"""
        self.skipped_sentences = 0
        self.processed_sentences = 0
        
        file_to_save = self.save_embeddings_path / f"{self.model_name}_{self.model_dim}.pth"
        if file_to_save.exists():
            print(f"[DEBUG] File {file_to_save} already exists.")
            return

        # Clean up GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Report initial GPU memory usage
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        # Load translation dataset for the current language
        translated_sentences = self._load_translated_dataset()
        
        try:
            if self.config.endpoint.lm_endpoint is not None:
                raise NotImplementedError("Endpoint support is not implemented yet.")
            else:
                # Process with local model
                self._process_with_local_model(translated_sentences)
                
            # Save extracted embeddings
            if not self.emb_per_object:
                self.save_avg_embed(translated_sentences["alias"])
        finally:
            # Close database connection
            self.con.close()
            
            # Clean up GPU memory after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Report final GPU memory usage
                print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Final GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        print(f"Language embeddings successfully extracted for {self.current_language}!")

    def _load_translated_dataset(self) -> Dataset:
        """Load dataset with translations for the current language"""

        dataset = load_dataset(self.config.dataset.language_data.dataset_name_hf, split="whole", token=self.config.hugging_face_save_repo.token)
        shuffled = dataset.shuffle(seed=self.config.common.seed)

        dataset_size = 79_000 if self.config.dataset.language_data.dataset_size is None \
            else min(79_000, self.config.dataset.language_data.dataset_size)

        random_subset = shuffled.select(range(dataset_size))
        return random_subset

    def _get_model_and_tokenizer(self):
        # Configure the model to output hidden states
        cache_path = Path.home() / ".cache/huggingface/transformers/models" / self.model_id
        configuration = AutoConfig.from_pretrained(
            self.model_id, cache_dir=cache_path, output_hidden_states=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache_path,
            use_fast=False
        )

        if self.model_name.startswith(("gpt")):
            tokenizer.pad_token = tokenizer.eos_token

        # Check for CUDA availability
        has_cuda = torch.cuda.is_available()
        
        # Set appropriate device
        if has_cuda:
            device = "cuda"
            # Clear GPU memory before loading model
            torch.cuda.empty_cache()

            max_memory = {
                k: f"{torch.cuda.get_device_properties(k).total_memory // 1024 ** 3}GB"
                for k in self.device
            }
        else:
            device = "cpu"
            # Disable quantization if no CUDA
            if self.config.model.use_quantization_8bit:
                self.config.model.use_quantization_8bit = False
                warnings.warn("CUDA is not available. Quantization has been disabled.", UserWarning)

        match self.config.model.torch_type:
            case "bfloat16" if not self.model_name.startswith(("gpt", "bert")) and not has_cuda:
                torch_type = torch.bfloat16
            case "float16" if not self.model_name.startswith(("gpt", "bert")):
                torch_type = torch.float16
            case _:
                torch_type = None


        model = AutoModel.from_pretrained(
            self.model_id,
            config=configuration,
            cache_dir=cache_path,
            load_in_8bit=self.config.model.use_quantization_8bit,
            device_map="auto" if has_cuda else None,  # Let transformers optimize device allocation
            max_memory=max_memory if has_cuda else None,
            torch_dtype=torch_type
        )
        
        device = torch.device(device)
        model.to(device)
        model.eval()
        
        # Apply optimization for inference
        # Optimize memory usage for inference
        if hasattr(model, "config") and getattr(model.config, "model_type", None) != "gpt2":
            # Don't use inference mode for GPT-2 as it can lead to issues
            torch.inference_mode(True)

        return model, tokenizer



    def _process_with_local_model(self, dataset: Dataset) -> None:
        """Process embeddings using local model with layer selection support"""
        model, tokenizer = self._get_model_and_tokenizer()

        # Process batch by batch
        pattern = r"\s+([^\w\s]+)(\s*)$"
        replacement = r"\1\2"
        
        # Collect all data for batch database insertion
        all_insert_data = []
        
        for i in tqdm(range(0, len(dataset), self.bs)):
            batch = dataset[i: i + self.bs]

            # flat sentences from the batch
            batch_flat_sentences = [
                re.sub(pattern, replacement, sentence)
                for sentences in batch["sentences"]
                for sentence in sentences
            ]

            # repeated aliases [alias_1, alias_2, alias_2, alias_3, ...]
            batch_aliases_repeated = list(chain.from_iterable([
                [word_data] * len(batch["sentences"][n]) for n, word_data in enumerate(batch["alias"])
            ]))

            embeddings, related_aliases = self.alias_embed(
                batch_flat_sentences, tokenizer, model, batch_aliases_repeated
            )

            # Prepare data for insertion but don't insert yet
            batch_insert_data = [(related_aliases[embedding_i], embedding.detach().cpu().numpy().tolist()) for embedding_i, embedding in enumerate(embeddings)]
            all_insert_data.extend(batch_insert_data)
            
            # Add explicit GPU memory cleanup at regular intervals
            if i % (self.bs * 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Perform batch database insertion every 1000 items to avoid memory buildup
            if len(all_insert_data) >= 1000:
                # batch_insert_data = [(alias, embedding.detach().cpu().numpy().tolist()) for (alias, embedding) in all_insert_data]
                self.con.executemany("INSERT INTO data VALUES (?, ?)", all_insert_data)
                self.con.commit()
                all_insert_data = []
        
        print(f"Number of skipped sentences: {self.skipped_sentences}.")
        print(f"Number of processed sentences: {self.processed_sentences}.")

        # Insert any remaining data
        if all_insert_data:
            self.con.executemany("INSERT INTO data VALUES (?, ?)", all_insert_data)
            self.con.commit()

    def alias_embed(
            self,
            batch: list,
            tokenizer: Any,
            model: Any,
            related_alias: list,
    ) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Extract embeddings with support for layer selection"""
        # Step 1: Tokenize input and run model
        tokens, hidden_states = self._tokenize_and_run_model(batch, tokenizer, model)
        
        # print(f"[DEBUG] len(hidden_states): {len(hidden_states)}, hidden_states[0].shape: {hidden_states[0].shape}")

        # Step 2: Select appropriate layers based on configuration
        selected_layers = self._select_layers(hidden_states) # torch.Size([6, 7, 768]) [layers], where shape(layer)=[bs,length,dim]
        
        # Step 3: Process each sentence and create embeddings
        embeddings, token_indices = self._process_embeddings(batch, tokens, selected_layers, related_alias, tokenizer)
        
        return embeddings, token_indices
    


    def _tokenize_and_run_model(self, batch: list, tokenizer: Any, model: Any) -> Tuple[Any, List[torch.Tensor]]:
        """Tokenize input and run model to get hidden states"""
        tokens = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            
        return tokens, outputs.hidden_states
    


    def _select_layers(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        """Select appropriate layers based on configuration"""
        if self.config.model.last_n_hidden_states is not None:
            total_layers = len(hidden_states)
            start_layer_idx = max(0, total_layers - self.last_n_hidden_states)
            if self.model_name == "opt-350m": return hidden_states[max(0, start_layer_idx - 1): -1]
            print("[DEBUG] Last several layers were used.")
            return hidden_states[start_layer_idx:]
            
        print(f"[DEBUG] The {self.config.model.specific_last_hidden_state} last layer was used.")
        if self.model_name == "opt-350m": return [hidden_states[-self.config.model.specific_last_hidden_state - 1]]
        return [hidden_states[-self.config.model.specific_last_hidden_state]]
        


    def _process_embeddings(
            self, 
            batch: list, 
            tokens: Any, 
            selected_layers: List[torch.Tensor], 
            related_alias: list,
            tokenizer: Any
    ) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Process each sentence and create embeddings"""
        embeddings = []
        aliases = []
        
        # NOTE: In other words it's iteration over batch
        for sentence_idx, alias in enumerate(related_alias):

            current_sentence = batch[sentence_idx]
            replaced_alias = alias.replace("_", " ").replace(".", "").strip() # "translation_abc ." ->  "translation abc"

            if replaced_alias.lower() not in current_sentence.lower():
                print(f"[DEBUG] {replaced_alias} is not in {current_sentence}")
                continue

            # Get tokens for the current sentence
            input_ids = tokens.input_ids[sentence_idx]
            mask = tokens.attention_mask[sentence_idx]
            
            # Identify tokens corresponding to the alias
            target_tokens_indices = self.map_word_to_token_ind(current_sentence, replaced_alias, tokenizer, mask.tolist(), input_ids.tolist())
            
            if target_tokens_indices != []:
                aliases.append(alias)
                # Extract embeddings based on layer selection configuration
                if len(selected_layers) > 1:
                    embedding = self._average_layers_embedding(selected_layers, sentence_idx, target_tokens_indices)
                else:
                    embedding = self._get_last_layer_embedding(selected_layers, sentence_idx, target_tokens_indices)
                    
                embeddings.append(embedding)
                self.processed_sentences += 1
            
            else:
                self.skipped_sentences += 1
            
        return embeddings, aliases
    


    def _average_layers_embedding(self, selected_layers: List[torch.Tensor], sentence_idx: int, words_mask: list) -> np.ndarray:
        """Get embedding by averaging across selected layers"""
        # print(f"[DEBUG](_average_layers_embedding) selected last layers: {len(selected_layers)}. Shape of the layer: {selected_layers[0].shape}")
        for i, layer in enumerate(selected_layers): print(f"Layer #{i} dim = {layer[sentence_idx].shape}.")
        layers_emb = torch.stack([layer[sentence_idx] for layer in selected_layers])
        avg_emb = torch.mean(layers_emb, dim=0)
        # Keep on GPU until the last moment
        return self.get_tokens_embedding(avg_emb, words_mask)
    


    def _get_last_layer_embedding(self, selected_layers: List[torch.Tensor], sentence_idx: int, words_mask: list) -> np.ndarray:
        """Get embedding from the last selected layer"""
        # print(f"[DEBUG](_get_last_layer_embedding) selected last layers: {len(selected_layers)}. Shape of the layer: {selected_layers[0].shape}")
        last_layer_emb = selected_layers[-1][sentence_idx]  # Keep as tensor, don't convert to numpy yet
        return self.get_tokens_embedding(last_layer_emb, words_mask)
    


    def map_word_to_token_ind(
            self, words_in_array: str, target_word: str, tokenizer: Any,
            words_mask: list, input_ids: list = None
    ) -> list:
        """
        Map tokens to words for accurate embedding extraction.
        Handles different tokenization schemes for various models.
        
        Args:
            words_in_array: The full sentence
            target_word: The target word/concept to find in the sentence
            tokenizer: The tokenizer instance
            words_mask: Attention mask for the tokens
            input_ids: Input token IDs (optional)
            
        Returns:
            list: Indices of tokens corresponding to the target word
        """
        # Get tokenizer type to determine appropriate mapping strategy

        # TODO: make tokenizer type as ENUM
        tokenizer_type = self._identify_tokenizer_type(tokenizer)
        
        # TODO: simplify code, because there are a lot of repetitions
        # Handle different tokenization schemes based on model type
        if tokenizer_type == "bert":
            return self._map_bert_tokens(words_in_array, target_word, tokenizer, words_mask)
        elif tokenizer_type in ["roberta", "deberta"]:
            return self._map_roberta_tokens(words_in_array, target_word, tokenizer, words_mask)
        elif tokenizer_type == "gpt2":
            return self._map_gpt2_tokens(words_in_array, target_word, tokenizer, words_mask)
        elif tokenizer_type in ["llama2", "llama3", "opt", "mixtral"]:
            return self._map_llama_family_tokens(words_in_array, target_word, tokenizer, words_mask)
        else:
            # Generic fallback method for other tokenizers
            print(f"[DEBUG] Generic tokenizer was chosen.")
            return self._map_generic_tokens(words_in_array, target_word, tokenizer, words_mask)
    
    def _identify_tokenizer_type(self, tokenizer: Any) -> str:
        """Identify the type of tokenizer based on its class name"""
        tokenizer_class = tokenizer.__class__.__name__.lower()

        if "bert" in tokenizer_class and "roberta" not in tokenizer_class and "deberta" not in tokenizer_class:
            return "bert"
        elif "roberta" in tokenizer_class:
            return "roberta"
        elif "deberta" in tokenizer_class:
            return "deberta"
        elif "gpt2" in tokenizer_class:
            return "gpt2"
        elif "llama" in tokenizer_class:
            # Differentiate between LLaMA versions
            if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size > 32000:
                return "llama3"  # LLaMA 3 has a larger vocabulary
            else:
                return "llama2"
        elif "opt" in tokenizer_class:
            return "opt"
        elif "mixtral" in tokenizer_class:
            return "mixtral"
        else:
            return "generic"
    
    def _map_bert_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
        """Map tokens for BERT-like models (BERT, DistilBERT)"""

        # BERT uses WordPiece tokenization with ## for subwords
        # tokens = tokenizer.tokenize(sentence.lower())
        # target_tokens = tokenizer.tokenize(target_word.lower())

        tokens = tokenizer.tokenize(sentence)
        target_tokens = tokenizer.tokenize(target_word)

        def lemmatize_bert_token(token: str) -> str:
            clean = token.replace("##", "").lower()
            parses = self.morph.parse(clean)
            return parses[0].normal_form if parses else clean

        tokens = [lemmatize_bert_token(t) for t in tokens]
        target_tokens = [lemmatize_bert_token(t) for t in target_tokens]


        # Find potential matches in the tokenized sentence
        matches = []
        for i in range(len(tokens) - len(target_tokens) + 1):
            # Check if we have a match for the first token
            if tokens[i].lower() == target_tokens[0].lower():
                match = True
                # Check if subsequent tokens match
                for j in range(1, len(target_tokens)):
                    if i + j >= len(tokens) or tokens[i + j].lower() != target_tokens[j].lower():
                        match = False
                        break
                if match:
                    matches.append(list(range(i, i + len(target_tokens))))
        
        # If we have matches, use the first one
        if matches:
            # An attention mask is typically a binary tensor with:
            # 1 for tokens that should be attended to (real tokens).
            # 0 for tokens that should be ignored (padding tokens).
            return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        

        print()
        print(f"[DEBUG](bert) fullback mapping was called on sentence {sentence.lower()}")
        print(f"[DEBUG](bert) fullback mapping was called on target_word {target_word.lower()}, {target_tokens}")
        print(f"[DEBUG](bert) fullback mapping was called on tokens {tokens}")
        print()

        # If no matches found, use a fallback method
        return self._map_fallback(tokens, words_mask)
    
    def _map_roberta_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
        """Map tokens for RoBERTa models which use byte-level BPE"""
        # RoBERTa uses byte-level BPE with Ġ at the start of tokens
        # tokens = tokenizer.tokenize(sentence.lower())
        # target_tokens = tokenizer.tokenize(" " + target_word.lower())  # Add space to match RoBERTa's prefix
        tokens = tokenizer.tokenize(sentence)
        target_tokens = tokenizer.tokenize(" " + target_word)  # Add space to match RoBERTa's prefix
        
        # Find where the target tokens appear in the sentence tokens
        matches = []
        for i in range(len(tokens) - len(target_tokens) + 1):
            match = True
            for j in range(len(target_tokens)):
                if i + j >= len(tokens) or tokens[i + j] != target_tokens[j]:
                    match = False
                    break
            if match:
                matches.append(list(range(i, i + len(target_tokens))))
        
        # If we have matches, use the first one
        if matches:
            return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
            
        # Try with alternative prefix (some RoBERTa variants)
        # target_tokens = tokenizer.tokenize(target_word.lower())
        target_tokens = tokenizer.tokenize(target_word)
        
        matches = []
        for i in range(len(tokens) - len(target_tokens) + 1):
            match = True
            for j in range(len(target_tokens)):
                if i + j >= len(tokens) or tokens[i + j] != target_tokens[j]:
                    match = False
                    break
            if match:
                matches.append(list(range(i, i + len(target_tokens))))
        
        # If we have matches, use the first one
        if matches:
            return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        
        # If no matches found, use fallback method
        return self._map_fallback(tokens, words_mask)
    
    # def _map_gpt2_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
    #     """Map tokens for GPT-2 models which use byte-level BPE"""
    #     # GPT-2 uses byte-level BPE, similar to RoBERTa but with different prefixes
    #     tokens = tokenizer.tokenize(sentence.lower())

    #     # TODO: Optimization is possible. We can take into account the possition of the target word in the sentence.
    #     # Try different prefix combinations for GPT-2

    #     target_word = target_word.lower()
    #     target_variations = [
    #         tokenizer.tokenize(target_word),
    #         tokenizer.tokenize(" " + target_word),
    #         tokenizer.tokenize("  " + target_word)
    #     ]
        
    #     for target_tokens in target_variations:
    #         matches = []
    #         for i in range(len(tokens) - len(target_tokens) + 1):
    #             match = True
    #             for j in range(len(target_tokens)):
    #                 if i + j >= len(tokens) or tokens[i + j] != target_tokens[j]:
    #                     match = False
    #                     break
    #             if match:
    #                 matches.append(list(range(i, i + len(target_tokens))))
            
    #         # If we have matches, use the first one
    #         if matches:
    #             return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        
    #     print()
    #     print(f"[DEBUG](_map_gpt2_tokens) fullback mapping was called on sentence: {sentence}")
    #     print(f"[DEBUG](_map_gpt2_tokens) fullback mapping was called on target_word: {target_word}")
    #     print(f"[DEBUG](_map_gpt2_tokens) fullback mapping was called on target_tokens [0]: {target_variations[0]}")
    #     print(f"[DEBUG](_map_gpt2_tokens) fullback mapping was called on target_tokens [1]: {target_variations[1]}")
    #     print(f"[DEBUG](_map_gpt2_tokens) fullback mapping was called on target_tokens [2]: {target_variations[2]}")
    #     print(f"[DEBUG](_map_gpt2_tokens) fullback mapping was called on tokens {tokens}")
    #     print()

    #     # If no matches found, use fallback method
    #     return self._map_fallback(tokens, words_mask)
    
    def _map_gpt2_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
        # """Map tokens for GPT-2 models with encoding fix, lemmatization, and punctuation normalization."""
        # import string
        # # Tokenize the sentence normally
        # tokens = tokenizer.tokenize(sentence.lower())

        # # A safer helper: try to fix misencoded tokens; if that fails, use the original token.
        # def fix_and_lemmatize(token: str) -> str:
        #     try:
        #         # Attempt to fix mis-decoded token by converting from Latin-1 to UTF-8.
        #         token_fixed = token.encode('latin-1').decode('utf-8')
        #     except (UnicodeEncodeError, UnicodeDecodeError):
        #         token_fixed = token  # Fallback if decoding fails
        #     # Remove GPT-2's prefix marker (e.g., "Ġ") if present.
        #     if token_fixed.startswith("Ġ"):
        #         token_fixed = token_fixed[1:]
        #     token_fixed = token_fixed.lower()
        #     parses = self.morph.parse(token_fixed)
        #     return parses[0].normal_form if parses else token_fixed

        # # Process sentence tokens with our safe function.
        # tokens = [fix_and_lemmatize(t) for t in tokens]

        # # Prepare target token variations with different preceding spaces.
        # target_word_lower = target_word.lower()
        # target_variations = [
        #     [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower)],
        #     [fix_and_lemmatize(t) for t in tokenizer.tokenize(" " + target_word_lower)],
        #     [fix_and_lemmatize(t) for t in tokenizer.tokenize("  " + target_word_lower)]
        # ]

        # # Iterate over each variation and try to match the token sequence.
        # for target_tokens in target_variations:
        #     # Remove empty tokens if any appear
        #     target_tokens = [t for t in target_tokens if t]
        #     matches = []
        #     for i in range(len(tokens) - len(target_tokens) + 1):
        #         match = True
        #         for j in range(len(target_tokens)):
        #             token_sentence = tokens[i + j]
        #             token_target = target_tokens[j]
        #             # Direct match; if not, try comparing after stripping punctuation
        #             if token_sentence != token_target:
        #                 if token_sentence.strip(string.punctuation) != token_target.strip(string.punctuation):
        #                     match = False
        #                     break
        #         if match:
        #             matches.append(list(range(i, i + len(target_tokens))))


        def adjust_target_word(target_word, sentence):
            temp_sentence = sentence.replace(".", "").replace(",", "").replace("!", "").replace("?", "")#.replace('"', "")

            splitted_target_word = target_word.split()
            splitted_sentence = temp_sentence.split()

            splitted_target_word_normal_form = [self.morph.parse(word)[0].normal_form for word in splitted_target_word]
            splitted_sentence_normal_form = [self.morph.parse(word)[0].normal_form for word in splitted_sentence]

            for i in range(len(splitted_sentence) - len(splitted_target_word) + 1):
                current_subsentence = splitted_sentence_normal_form[i: i + len(splitted_target_word)]
                if current_subsentence == splitted_target_word_normal_form:
                    return " ".join(splitted_sentence[i: i + len(splitted_target_word)])

            return target_word

        sentence = sentence.lower()
        sentence = sentence.replace(".", "")

        target_word = adjust_target_word(target_word.lower(), sentence)


        import string
        # Tokenize the sentence normally
        tokens = tokenizer.tokenize(sentence)

        # A safer helper: try to fix misencoded tokens; if that fails, use the original token.
        def fix_and_lemmatize(token: str) -> str:
            try:
                # Attempt to fix mis-decoded token by converting from Latin-1 to UTF-8.
                token_fixed = token.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                token_fixed = token  # Fallback if decoding fails
            # Remove GPT-2's prefix marker (e.g., "Ġ") if present.
            if token_fixed.startswith("Ġ"):
                token_fixed = token_fixed[1:]
            token_fixed = token_fixed.lower()
            parses = self.morph.parse(token_fixed)
            return parses[0].normal_form if parses else token_fixed

        # Process sentence tokens with our safe function.
        tokens = [fix_and_lemmatize(t) for t in tokens]

        # Prepare target token variations with different preceding spaces.
        target_word_lower = target_word.lower()
        target_variations = [
            tokenizer.tokenize(target_word_lower),

            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower)],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '"')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '",')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '".')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '"!')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '"?')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '?')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '!')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '!')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + '.')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(target_word_lower + ',')],

            [fix_and_lemmatize(t) for t in tokenizer.tokenize(" " + target_word_lower)],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(" " + target_word_lower + '"')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize(" " + target_word_lower + '",')],

            [fix_and_lemmatize(t) for t in tokenizer.tokenize("  " + target_word_lower)],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize("  " + target_word_lower + '"')],
            [fix_and_lemmatize(t) for t in tokenizer.tokenize("  " + target_word_lower + '",')],

        ]

        for target_tokens in target_variations:
            matches = []
            for i in range(len(tokens) - len(target_tokens) + 1):
                match = True
                for j in range(len(target_tokens)):
                    if i + j >= len(tokens) or tokens[i + j] != target_tokens[j]:
                        match = False
                        break
                if match:
                    matches.append(list(range(i, i + len(target_tokens))))
                    
            if matches:
                return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]

        # Debug logging if no match is found.
        print()
        print(f"[DEBUG](_map_gpt2_tokens) fallback mapping was called on sentence: {sentence}")
        print(f"[DEBUG](_map_gpt2_tokens) fallback mapping was called on target_word: {target_word_lower}")
        for i in range(len(target_variations)):
            print(f"[DEBUG](_map_gpt2_tokens) fallback mapping was called on target_tokens [{i}]: {target_variations[i]}")
        # print(f"[DEBUG](_map_gpt2_tokens) fallback mapping was called on target_tokens [1]: {[fix_and_lemmatize(t) for t in tokenizer.tokenize(' ' + target_word_lower)]}")
        # print(f"[DEBUG](_map_gpt2_tokens) fallback mapping was called on target_tokens [2]: {[fix_and_lemmatize(t) for t in tokenizer.tokenize('  ' + target_word_lower)]}")
        # print(f"[DEBUG](_map_gpt2_tokens) fallback mapping was called on tokens: {tokens}")
        print()

        # Use fallback mapping if no match is found.
        return self._map_fallback(tokens, words_mask)



    def _map_llama_family_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
        """Map tokens for LLaMA, OPT, Mixtral models which use SentencePiece or similar tokenizers"""
        # Check specific LLaMA version
        tokenizer_type = self._identify_tokenizer_type(tokenizer)
        
        # Use LLaMA 3 specific mapping if identified
        if tokenizer_type == "llama3":
            return self._map_llama3_tokens(sentence, target_word, tokenizer, words_mask)
        
        # Standard handling for LLaMA 2, OPT, Mixtral, etc.
        # tokens = tokenizer.tokenize(sentence.lower())
        tokens = tokenizer.tokenize(sentence)
        
        # Check for common SentencePiece prefixes in tokens to adjust strategy
        has_underscore_prefix = any("▁" in str(t) for t in tokens[:10] if isinstance(t, str))
        space_prefix = "▁" if has_underscore_prefix else " "
        
        # Try different variations for LLaMA family models with appropriate prefixes
        # target_word = target_word.lower()
        target_word = target_word
        target_variations = [
            tokenizer.tokenize(target_word),
            tokenizer.tokenize(space_prefix + target_word),
            tokenizer.tokenize(space_prefix + space_prefix + target_word)
        ]
        
        # Filter out empty variations
        target_variations = [v for v in target_variations if v]
        
        for target_tokens in target_variations:
            matches = []
            for i in range(len(tokens) - len(target_tokens) + 1):
                match = True
                for j in range(len(target_tokens)):
                    if i + j >= len(tokens) or tokens[i + j] != target_tokens[j]:
                        match = False
                        break
                if match:
                    matches.append(list(range(i, i + len(target_tokens))))
            
            # If we have matches, use the first one
            if matches:
                return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        
        # Try using offset mapping if available
        try:
            # Some tokenizers provide character offset information
            encoding = tokenizer(sentence.lower(), return_offsets_mapping=True, add_special_tokens=False)
            if 'offset_mapping' in encoding:
                target_lower = target_word.lower()
                sentence_lower = sentence.lower()
                offsets = encoding['offset_mapping']
                
                # Find all occurrences of the target in the sentence
                start_idx = sentence_lower.find(target_lower)
                matches = []
                
                while start_idx != -1:
                    end_idx = start_idx + len(target_lower)
                    token_indices = []
                    
                    # Find tokens that overlap with this occurrence
                    for i, (token_start, token_end) in enumerate(offsets):
                        if token_end > start_idx and token_start < end_idx:
                            token_indices.append(i)
                    
                    if token_indices:
                        matches.append(token_indices)
                        
                    start_idx = sentence_lower.find(target_lower, start_idx + 1)
                
                if matches:
                    return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        except Exception:
            # If this approach fails, continue to next method
            pass
        

        print()
        print(f"[DEBUG] fullback mapping was called on sentence {sentence}")
        print(f"[DEBUG] fullback mapping was called on target_word {target_word}")
        print(f"[DEBUG] fullback mapping was called on tokens {tokens}")
        print()
        # If no matches found, use the generic fallback method
        return self._map_generic_tokens(sentence, target_word, tokenizer, words_mask)
    
    def _map_llama3_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
        """Map tokens specifically for LLaMA 3 which has some tokenization differences from LLaMA 2"""
        # tokens = tokenizer.tokenize(sentence.lower())
        tokens = tokenizer.tokenize(sentence)
        
        # target_word = target_word.lower()
        # Try different prefix variations specific to LLaMA 3
        target_variations = [
            tokenizer.tokenize(target_word),
            tokenizer.tokenize(" " + target_word),
            tokenizer.tokenize("  " + target_word),
            # Add some LLaMA 3 specific variations if needed
            tokenizer.tokenize("<0x20>" + target_word) if "<0x20>" in str(tokens) else []
        ]
        
        # Filter out empty variations
        target_variations = [v for v in target_variations if v]
        
        for target_tokens in target_variations:
            matches = []
            for i in range(len(tokens) - len(target_tokens) + 1):
                match = True
                for j in range(len(target_tokens)):
                    if i + j >= len(tokens) or tokens[i + j] != target_tokens[j]:
                        match = False
                        break
                if match:
                    matches.append(list(range(i, i + len(target_tokens))))
            
            # If we have matches, use the first one
            if matches:
                return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        
        # Try with character-level mapping specific to LLaMA 3
        try:
            # Use character offset mappings if available in tokenizer
            target_lower = target_word.lower()
            sentence_lower = sentence.lower()
            
            encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
            if 'offset_mapping' in encoding:
                offsets = encoding['offset_mapping']
                
                # Find occurrences of target in sentence
                start_idx = sentence_lower.find(target_lower)
                matches = []
                
                while start_idx != -1:
                    end_idx = start_idx + len(target_lower)
                    token_indices = []
                    
                    # Find tokens that overlap with this occurrence
                    for i, (token_start, token_end) in enumerate(offsets):
                        if token_end > start_idx and token_start < end_idx:
                            token_indices.append(i)
                    
                    if token_indices:
                        matches.append(token_indices)
                        
                    start_idx = sentence_lower.find(target_lower, start_idx + 1)
                
                if matches:
                    # Use the first match if available
                    return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
        except Exception:
            # Fallback to generic method if this fails
            pass
        
        # If no matches found, use fallback method
        return self._map_fallback(tokens, words_mask)
    
    def _map_generic_tokens(self, sentence: str, target_word: str, tokenizer: Any, words_mask: list) -> list:
        """Generic mapping method for unknown tokenizers"""
        try:
            # First try with word-level matching
            tokens = tokenizer.tokenize(sentence)
            # Try with direct tokenization
            target_tokens = tokenizer.tokenize(target_word)
            
            # Find potential matches in the tokenized sentence
            matches = []
            for i in range(len(tokens) - len(target_tokens) + 1):
                match = True
                for j in range(len(target_tokens)):
                    if i + j >= len(tokens) or tokens[i + j].lower() != target_tokens[j].lower():
                        match = False
                        break
                if match:
                    matches.append(list(range(i, i + len(target_tokens))))
            
            # If we have matches, use the first one
            if matches:
                return [idx for idx in matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
            
            # Try alternative method: find all token positions where the target word might appear
            target_lower = target_word.lower()
            sentence_lower = sentence.lower()
            
            # Find all occurrences of the target in the sentence
            start_positions = []
            start_idx = sentence_lower.find(target_lower)
            while start_idx != -1:
                start_positions.append(start_idx)
                start_idx = sentence_lower.find(target_lower, start_idx + 1)
            
            if start_positions:
                # Map character positions to token positions
                char_to_token = {}
                char_idx = 0
                
                for token_idx, token in enumerate(tokens):
                    # Approximate character length of the token
                    token_len = len(token.replace("##", "").replace("Ġ", "").replace("▁", ""))
                    for _ in range(token_len):
                        char_to_token[char_idx] = token_idx
                        char_idx += 1
                
                # Collect token indices for each occurrence
                token_matches = []
                for start_pos in start_positions:
                    token_indices = set()
                    for char_pos in range(start_pos, min(start_pos + len(target_word), len(sentence))):
                        if char_pos in char_to_token:
                            token_indices.add(char_to_token[char_pos])
                    if token_indices:
                        token_matches.append(sorted(token_indices))
                
                # Use the first match if available
                if token_matches:
                    return [idx for idx in token_matches[0] if idx < len(words_mask) and words_mask[idx] == 1]
            
            # If all else fails, use fallback method
            return self._map_fallback(tokens, words_mask)
        except Exception as e:
            # In case of any error, use the fallback method
            print(f"Error in generic token mapping: {e}")
            return self._map_fallback(tokens if 'tokens' in locals() else [], words_mask)
    
    def _map_fallback(self, tokens: list, words_mask: list) -> list:
        """Fallback method when token mapping fails"""
        # Use all tokens with attention mask = 1
        print(f"[DEBUG] fullback mapping was called on {tokens}")
        # return [i for i, m in enumerate(words_mask) if i < len(tokens) and m == 1]
        return [] #[i for i, m in enumerate(words_mask) if i < len(tokens) and m == 1]



    def get_tokens_embedding(
            self,
            embeddings_to_add: Union[np.ndarray, torch.Tensor],
            tokens_indices: list,
    ) -> np.ndarray:
        """Extract token embeddings for specific tokens"""
        # Get embeddings for tokens of interest
        token_embeddings = embeddings_to_add[tokens_indices]
        # Average embeddings for all tokens
        avg_embedding = torch.mean(token_embeddings, axis=0)
        return avg_embedding



    def save_avg_embed(self, aliases: Dataset) -> None:
        # """Save averaged embeddings for each concept"""
        # concepts = aliases["concept"]
        
        # # Create directory for averaged embeddings
        # avg_dir = self.alias_emb_dir / "averaged_embeddings" / self.dataset_name
        # os.makedirs(avg_dir, exist_ok=True)
        
        # # Get unique concepts and their embeddings
        # unique_aliases = {}
        # self.con.execute("SELECT alias, embedding FROM data")
        # rows = self.con.fetchall()
        
        # for alias, embedding in rows:
        #     if alias not in unique_aliases:
        #         unique_aliases[alias] = []
        #     unique_aliases[alias].append(embedding)
        
        # # Calculate average embeddings
        # for alias, embeddings in unique_aliases.items():
        #     avg_emb = np.mean(embeddings, axis=0)
        #     # Normalize embedding
        #     avg_emb = avg_emb / np.linalg.norm(avg_emb)
        #     # Save embedding
        #     torch.save(avg_emb, avg_dir / f"{alias}.pth")
        
        # print(f"Saved averaged embeddings for {len(unique_aliases)} concepts")
        
        # # Save all embeddings in one file
        # all_embeddings = {}
        # for alias, avg_emb in unique_aliases.items():
        #     all_embeddings[alias] = avg_emb
            
        # torch.save(all_embeddings, avg_dir / f"{self.model_name}_{self.model_dim}.pth") 

        avg_embeddings, final_alias = [], []
        for i in aliases:
            query = f"SELECT embedding FROM data WHERE alias = '{i}'"
            result = self.con.execute(query).fetchall()
            if result:
                result = np.concatenate(result, axis=0)
                avg_embeddings.append(np.mean(result, axis=0))
                final_alias.append(i)
        print(result.shape)
        # torch.save({"dico": final_alias, "vectors": torch.from_numpy(np.array(avg_embeddings))},
        #            self.save_embeddings_path / f"{self.model_name}_{self.model_dim}.pth")

        embegginds_dim = result.shape[-1]
        object_to_save = {"dico": final_alias, "vectors": torch.from_numpy(np.array(avg_embeddings))}
        # save_path = self.save_embeddings_path / f"{self.model_name}_{self.model_dim}.pth"
        save_path = self.save_embeddings_path / f"{self.model_name}_{embegginds_dim}.pth"
        self.file_saver.save(object_to_save, save_path)