# data.py
"""
Data loading et collators pour SLGA

Inclut:
- Chargement de datasets HuggingFace
- Tokenization
- Collators pour local-only et local+global (heuristiques)
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import numpy as np
from src.dataset_cleaner import CleanedDataset


def get_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Charge un tokenizer HuggingFace.
    
    Args:
        tokenizer_name: Nom du tokenizer (ex: "gpt2")
    
    Returns:
        tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # S'assurer qu'on a un pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


# src/data.py
from datasets import load_dataset, load_from_disk
import os

def load_text_dataset(dataset_name: str, subset: str = None, split: str = "train"):
    """
    Charge un dataset texte depuis HuggingFace Hub OU depuis disque local.
    
    Args:
        dataset_name: Nom HuggingFace (ex: "openwebtext") OU path local (ex: "data/mixed")
        subset: Subset du dataset (ignoré si local)
        split: Split à charger
    """
    # ============================================================================
    # DÉTECTION: Local vs HuggingFace
    # ============================================================================
    if os.path.exists(dataset_name):
        # Dataset LOCAL (sauvegardé avec save_to_disk)
        print(f"📁 Loading LOCAL dataset from: {dataset_name}")
        
        try:
            # Charger depuis disque
            dataset = load_from_disk(dataset_name)
            
            # Si le dataset a des splits, extraire le bon
            if isinstance(dataset, dict):  # DatasetDict
                if split in dataset:
                    return dataset[split]
                else:
                    available = list(dataset.keys())
                    raise ValueError(
                        f"Split '{split}' not found in local dataset. "
                        f"Available: {available}"
                    )
            else:
                # Dataset simple sans splits
                return dataset
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to load local dataset from '{dataset_name}': {e}"
            )
    
    else:
        # Dataset HUGGINGFACE HUB
        print(f"🌐 Loading HuggingFace dataset: {dataset_name}")
        
        if subset:
            print(f"   Subset: {subset}")
            dataset = load_dataset(dataset_name, subset, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        return dataset

class CollatorLocal:
    """
    Collator pour mode local-only (pas de landmarks heuristiques).
    
    Utilisé quand learned_landmarks=True (le modèle sélectionne lui-même).
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate un batch d'exemples.
        
        Args:
            examples: Liste de dicts avec clé 'text' (ou text_key)
        
        Returns:
            batch: Dict avec 'input_ids', 'labels'
        """
        # Extraire textes
        texts = [ex[self.text_key] for ex in examples]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,  # +1 pour shift labels
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"]  # (B, L+1)

        # Tronquer AVANT de créer labels pour éviter bugs
        input_ids = input_ids[:, :self.max_length + 1]  # Garder L+1 pour shift

        # Labels: shift de 1 position (prédire token suivant)
        # input_ids: [tok0, tok1, tok2, ..., tokL]
        # labels:    [tok1, tok2, ..., tokL, PAD]
        labels = input_ids[:, 1:].clone()  # (B, L) - déjà la bonne taille!
        input_ids = input_ids[:, :-1]  # (B, L)

        # 🔧 CRITICAL FIX: Masquer les tokens de padding avec -100
        # APRÈS avoir créé labels de la bonne taille
        pad_mask = (labels == self.tokenizer.pad_token_id)
        labels[pad_mask] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class CollatorLocalGlobal:
    """
    Collator avec landmarks globaux heuristiques.
    
    Stratégies:
    - "regular": Landmarks régulièrement espacés (tous les N tokens)
    - "paragraph": Débuts de paragraphes (détection \n\n)
    - "random": Positions aléatoires (baseline)
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        global_every: int = 128,
        max_global: int = 64,
        strategy: str = "regular",
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.global_every = global_every
        self.max_global = max_global
        self.strategy = strategy
        self.text_key = text_key
    
    def _select_landmarks_regular(self, length: int) -> List[int]:
        """
        Sélection régulière: exactement max_global landmarks uniformément espacés.

        FIX: Utilise linspace pour garantir exactement max_global landmarks,
        au lieu de range() qui peut créer G+1 landmarks.
        """
        import torch
        # Utiliser linspace pour avoir EXACTEMENT max_global landmarks
        positions = torch.linspace(0, length - 1, self.max_global).long().tolist()
        return positions
    
    def _select_landmarks_random(self, length: int) -> List[int]:
        """Sélection aléatoire"""
        num_landmarks = min(self.max_global, length)
        landmarks = sorted(np.random.choice(length, num_landmarks, replace=False).tolist())
        return landmarks
    
    def _select_landmarks_paragraph(self, text: str, tokens: List[int]) -> List[int]:
        """
        Sélection basée sur paragraphes.
        
        Détecte débuts de paragraphes (après \n\n) et mappe sur positions de tokens.
        """
        # Trouver positions des \n\n dans le texte original
        paragraphs = text.split("\n\n")
        
        # Approximation grossière: tokenizer chaque paragraphe et cumuler longueurs
        landmarks = [0]  # Toujours inclure début
        cumulative_len = 0
        
        for para in paragraphs[:-1]:  # Exclure dernier (pas de début après)
            para_tokens = self.tokenizer.encode(para, add_special_tokens=False)
            cumulative_len += len(para_tokens)
            if cumulative_len < len(tokens):
                landmarks.append(cumulative_len)
        
        # Limiter et trier
        landmarks = sorted(set(landmarks))[:self.max_global]
        
        return landmarks
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate avec landmarks globaux.
        
        Returns:
            batch: Dict avec 'input_ids', 'labels', 'cache_global_ids'
        """
        texts = [ex[self.text_key] for ex in examples]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"]
        B, L_plus = input_ids.shape

        # Tronquer AVANT de créer labels
        input_ids = input_ids[:, :self.max_length + 1]

        # Labels: shift de 1 position
        labels = input_ids[:, 1:].clone()  # (B, L)
        input_ids = input_ids[:, :-1]  # (B, L)

        # 🔧 CRITICAL FIX: Masquer les tokens de padding avec -100
        pad_mask = (labels == self.tokenizer.pad_token_id)
        labels[pad_mask] = -100

        # Sélectionner landmarks pour chaque exemple
        cache_global_ids_list = []
        
        for i, text in enumerate(texts):
            L = self.max_length
            
            if self.strategy == "regular":
                landmarks = self._select_landmarks_regular(L)
            elif self.strategy == "random":
                landmarks = self._select_landmarks_random(L)
            elif self.strategy == "paragraph":
                tokens_i = input_ids[i].tolist()
                landmarks = self._select_landmarks_paragraph(text, tokens_i)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Padding si moins de max_global
            while len(landmarks) < self.max_global:
                landmarks.append(0)  # Pad avec position 0
            
            cache_global_ids_list.append(landmarks[:self.max_global])
        
        cache_global_ids = torch.tensor(cache_global_ids_list, dtype=torch.long)  # (B, G)

        # ✅ CRITICAL FIX Bug #10: Retourner les POSITIONS, pas les tokens
        # Le modèle attend des positions pour faire gather dans forward()
        # Retourner les tokens corrompait le système de landmarks heuristiques

        return {
            "input_ids": input_ids,
            "labels": labels,
            "cache_global_ids": cache_global_ids,  # POSITIONS (pas tokens!)
        }


class CollatorWithTFIDF:
    """
    Collator avancé avec sélection TF-IDF.
    
    Utilise sklearn pour scorer des spans de tokens et sélectionner les plus informatifs.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_global: int = 64,
        span_length: int = 8,
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_global = max_global
        self.span_length = span_length
        self.text_key = text_key
        
        # TF-IDF (nécessite sklearn)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        except ImportError:
            print("Warning: sklearn not found, TF-IDF collator not available")
            self.tfidf = None
    
    def _select_landmarks_tfidf(self, text: str, tokens: List[int]) -> List[int]:
        """
        Sélection basée sur TF-IDF des spans.
        
        Divise la séquence en spans de longueur span_length,
        score chaque span avec TF-IDF, sélectionne top-K.
        """
        if self.tfidf is None:
            # Fallback: régulier
            return list(range(0, len(tokens), len(tokens) // self.max_global))[:self.max_global]
        
        # Diviser en spans
        num_spans = len(tokens) // self.span_length
        spans = []
        span_starts = []
        
        for i in range(num_spans):
            start = i * self.span_length
            end = start + self.span_length
            span_tokens = tokens[start:end]
            span_text = self.tokenizer.decode(span_tokens)
            spans.append(span_text)
            span_starts.append(start)
        
        if not spans:
            return [0]
        
        # TF-IDF scoring
        try:
            # Fit et transform (sur ce document)
            tfidf_matrix = self.tfidf.fit_transform(spans)
            scores = tfidf_matrix.sum(axis=1).A1  # (num_spans,)
            
            # Top-K spans
            num_landmarks = min(self.max_global, len(scores))
            top_indices = np.argsort(scores)[-num_landmarks:][::-1]
            
            # Mapper sur positions de tokens (début de chaque span)
            landmarks = sorted([span_starts[i] for i in top_indices])
            
        except Exception:
            # En cas d'erreur, fallback régulier
            landmarks = list(range(0, len(tokens), len(tokens) // self.max_global))[:self.max_global]
        
        return landmarks
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex[self.text_key] for ex in examples]
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = self.tokenizer.pad_token_id
        
        input_ids = input_ids[:, :self.max_length]
        labels = labels[:, :self.max_length]
        
        # Sélection TF-IDF
        cache_global_ids_list = []
        
        for i, text in enumerate(texts):
            tokens_i = input_ids[i].tolist()
            landmarks = self._select_landmarks_tfidf(text, tokens_i)
            
            # Padding
            while len(landmarks) < self.max_global:
                landmarks.append(0)
            
            cache_global_ids_list.append(landmarks[:self.max_global])
        
        cache_global_ids = torch.tensor(cache_global_ids_list, dtype=torch.long)
        cache_global_tokens = torch.gather(
            input_ids, dim=1, index=cache_global_ids.clamp(0, self.max_length - 1)
        )
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "cache_global_ids": cache_global_tokens,
        }


def test_collators():
    """Test des collators"""
    print("=== Test Collators ===")
    
    tokenizer = get_tokenizer("gpt2")
    
    # Données fictives
    examples = [
        {"text": "This is a test sentence. " * 50},
        {"text": "Another example text. " * 50},
    ]
    
    # Test CollatorLocal
    print("\n1. CollatorLocal")
    collator_local = CollatorLocal(tokenizer, max_length=128)
    batch = collator_local(examples)
    
    print(f"input_ids: {batch['input_ids'].shape}")
    print(f"labels: {batch['labels'].shape}")
    assert batch["input_ids"].shape == (2, 128)
    assert batch["labels"].shape == (2, 128)
    print("✓ CollatorLocal OK")
    
    # Test CollatorLocalGlobal
    print("\n2. CollatorLocalGlobal (regular)")
    collator_global = CollatorLocalGlobal(
        tokenizer, max_length=128, global_every=32, max_global=8, strategy="regular"
    )
    batch = collator_global(examples)
    
    print(f"input_ids: {batch['input_ids'].shape}")
    print(f"labels: {batch['labels'].shape}")
    print(f"cache_global_ids: {batch['cache_global_ids'].shape}")
    assert batch["cache_global_ids"].shape == (2, 8)
    print("✓ CollatorLocalGlobal OK")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_collators()
