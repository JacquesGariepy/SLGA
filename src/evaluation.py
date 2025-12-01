#!/usr/bin/env python3
"""
Module d'évaluation de la qualité des générations SLGA.

Métriques incluses:
- Perplexité (token-level et sequence-level)
- Diversité (distinct n-grams, entropie lexicale, type-token ratio)
- Répétition (taux de répétition n-grams, boucles de répétition)
- Cohérence (cohérence sémantique via embeddings)
- Fluence (probabilités de transition, détection de charabia)

Usage:
    from src.evaluation import GenerationEvaluator

    evaluator = GenerationEvaluator(model, tokenizer)
    metrics = evaluator.evaluate(prompt, generated_text)
    evaluator.generate_report(output_path)
"""

from __future__ import annotations
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn.functional as F
import numpy as np


# ======================================================================
# Data Classes for Metrics
# ======================================================================

@dataclass
class PerplexityMetrics:
    """Métriques de perplexité."""
    token_perplexity: float = 0.0      # Perplexité moyenne par token
    sequence_perplexity: float = 0.0   # Perplexité de la séquence complète
    loss: float = 0.0                  # Cross-entropy loss
    min_token_ppl: float = 0.0         # PPL min (meilleur token)
    max_token_ppl: float = 0.0         # PPL max (pire token)
    std_token_ppl: float = 0.0         # Écart-type PPL
    confident_tokens_ratio: float = 0.0  # % tokens avec PPL < 10

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class DiversityMetrics:
    """Métriques de diversité lexicale."""
    distinct_1: float = 0.0    # Ratio unigrams distincts
    distinct_2: float = 0.0    # Ratio bigrams distincts
    distinct_3: float = 0.0    # Ratio trigrams distincts
    distinct_4: float = 0.0    # Ratio 4-grams distincts
    type_token_ratio: float = 0.0       # TTR = vocab_unique / total_tokens
    log_type_token_ratio: float = 0.0   # LogTTR (corrigé pour la longueur)
    lexical_entropy: float = 0.0        # Entropie de Shannon
    vocabulary_richness: float = 0.0    # Mesure de Yule's K
    hapax_legomena_ratio: float = 0.0   # % mots apparaissant une seule fois

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RepetitionMetrics:
    """Métriques de répétition."""
    rep_1: float = 0.0    # Taux répétition unigrams
    rep_2: float = 0.0    # Taux répétition bigrams
    rep_3: float = 0.0    # Taux répétition trigrams
    rep_4: float = 0.0    # Taux répétition 4-grams
    max_consecutive_repeat: int = 0     # Plus longue chaîne de répétition
    loop_detected: bool = False         # Boucle de répétition détectée
    loop_pattern: Optional[str] = None  # Pattern de boucle détecté
    word_repetition_rate: float = 0.0   # % mots répétés consécutivement
    phrase_repetition_rate: float = 0.0 # % phrases répétées

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoherenceMetrics:
    """Métriques de cohérence et fluence."""
    avg_token_probability: float = 0.0  # Probabilité moyenne des tokens
    min_token_probability: float = 0.0  # Probabilité minimale
    low_prob_token_ratio: float = 0.0   # % tokens avec prob < 0.01
    gibberish_score: float = 0.0        # Score de charabia (0-1, 1=charabia)
    coherence_score: float = 0.0        # Score de cohérence (0-1, 1=cohérent)

    # Détection de problèmes spécifiques
    has_broken_words: bool = False      # Mots cassés/incomplets
    has_mixed_languages: bool = False   # Mélange de langues
    has_special_char_spam: bool = False # Spam de caractères spéciaux

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationQualityMetrics:
    """Métriques globales de qualité."""
    overall_score: float = 0.0          # Score global (0-100)
    quality_grade: str = "F"            # Note (A, B, C, D, F)

    perplexity: PerplexityMetrics = field(default_factory=PerplexityMetrics)
    diversity: DiversityMetrics = field(default_factory=DiversityMetrics)
    repetition: RepetitionMetrics = field(default_factory=RepetitionMetrics)
    coherence: CoherenceMetrics = field(default_factory=CoherenceMetrics)

    # Métadonnées
    prompt: str = ""
    generated_text: str = ""
    generated_only: str = ""  # Texte généré sans le prompt
    num_tokens: int = 0
    num_words: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0

    # Diagnostics
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "quality_grade": self.quality_grade,
            "perplexity": self.perplexity.to_dict(),
            "diversity": self.diversity.to_dict(),
            "repetition": self.repetition.to_dict(),
            "coherence": self.coherence.to_dict(),
            "metadata": {
                "prompt": self.prompt,
                "generated_text": self.generated_text,
                "generated_only": self.generated_only,
                "num_tokens": self.num_tokens,
                "num_words": self.num_words,
                "generation_time": self.generation_time,
                "tokens_per_second": self.tokens_per_second,
            },
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


# ======================================================================
# Generation Evaluator
# ======================================================================

class GenerationEvaluator:
    """
    Évaluateur complet de la qualité des générations.

    Calcule des métriques de perplexité, diversité, répétition et cohérence.
    Génère des rapports détaillés et des logs pour TensorBoard.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cuda",
    ):
        """
        Initialise l'évaluateur.

        Args:
            model: Modèle SLGA pour calcul de perplexité
            tokenizer: Tokenizer pour décodage
            device: Device pour les calculs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Historique des évaluations
        self.evaluation_history: List[GenerationQualityMetrics] = []

        # Patterns pour détection de problèmes
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile les patterns regex pour détection de problèmes."""
        # Caractères spéciaux répétés
        self.special_char_pattern = re.compile(r'[^\w\s]{3,}')
        # Mots cassés (CamelCase intempestif)
        self.broken_word_pattern = re.compile(r'[a-z][A-Z][a-z]')
        # Caractères non-ASCII
        self.non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
        # Chiffres isolés
        self.isolated_numbers_pattern = re.compile(r'\s\d{3,}\s')

    def evaluate(
        self,
        prompt: str,
        generated_text: str,
        generation_time: float = 0.0,
        compute_perplexity: bool = True,
    ) -> GenerationQualityMetrics:
        """
        Évalue la qualité d'une génération.

        Args:
            prompt: Texte du prompt
            generated_text: Texte complet (prompt + génération)
            generation_time: Temps de génération en secondes
            compute_perplexity: Si True, calcule la perplexité (nécessite model)

        Returns:
            GenerationQualityMetrics avec toutes les métriques
        """
        metrics = GenerationQualityMetrics()
        metrics.prompt = prompt
        metrics.generated_text = generated_text
        metrics.generation_time = generation_time

        # Extraire la partie générée seulement
        if generated_text.startswith(prompt):
            generated_only = generated_text[len(prompt):].strip()
        else:
            generated_only = generated_text

        metrics.generated_only = generated_only

        # Tokenizer pour métriques
        if self.tokenizer:
            tokens = self.tokenizer.encode(generated_only)
            metrics.num_tokens = len(tokens)
        else:
            # Estimation basique
            tokens = generated_only.split()
            metrics.num_tokens = len(tokens)

        words = generated_only.split()
        metrics.num_words = len(words)

        if generation_time > 0 and metrics.num_tokens > 0:
            metrics.tokens_per_second = metrics.num_tokens / generation_time

        # Calcul des métriques
        if compute_perplexity and self.model and self.tokenizer:
            metrics.perplexity = self._compute_perplexity(generated_text)

        metrics.diversity = self._compute_diversity(generated_only)
        metrics.repetition = self._compute_repetition(generated_only, tokens if isinstance(tokens, list) else None)
        metrics.coherence = self._compute_coherence(generated_only, tokens if not isinstance(tokens, list) else None)

        # Score global et diagnostics
        metrics.overall_score, metrics.quality_grade = self._compute_overall_score(metrics)
        metrics.issues, metrics.recommendations = self._generate_diagnostics(metrics)

        # Ajouter à l'historique
        self.evaluation_history.append(metrics)

        return metrics

    def _compute_perplexity(self, text: str) -> PerplexityMetrics:
        """Calcule les métriques de perplexité."""
        metrics = PerplexityMetrics()

        if not self.model or not self.tokenizer:
            return metrics

        try:
            self.model.eval()

            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            input_ids = input_ids.to(self.device)

            if input_ids.size(1) < 2:
                return metrics

            with torch.no_grad():
                logits = self.model(input_ids)  # (B, L, V)

                # Shift pour causal LM
                shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
                shift_labels = input_ids[:, 1:].contiguous()   # (B, L-1)

                # Loss par token
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)  # (B, L-1)

                # Perplexité par token
                token_ppls = torch.exp(-token_log_probs)  # (B, L-1)

                # Métriques
                metrics.loss = -token_log_probs.mean().item()
                metrics.sequence_perplexity = torch.exp(torch.tensor(metrics.loss)).item()
                metrics.token_perplexity = token_ppls.mean().item()
                metrics.min_token_ppl = token_ppls.min().item()
                metrics.max_token_ppl = token_ppls.max().item()
                metrics.std_token_ppl = token_ppls.std().item()

                # Ratio de tokens confiants (PPL < 10)
                confident_mask = token_ppls < 10
                metrics.confident_tokens_ratio = confident_mask.float().mean().item()

        except Exception as e:
            warnings.warn(f"Erreur calcul perplexité: {e}")

        return metrics

    def _compute_diversity(self, text: str) -> DiversityMetrics:
        """Calcule les métriques de diversité lexicale."""
        metrics = DiversityMetrics()

        if not text.strip():
            return metrics

        # Tokenisation simple par mots
        words = text.lower().split()

        if len(words) == 0:
            return metrics

        # N-grams
        def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

        # Distinct-N metrics
        for n in [1, 2, 3, 4]:
            ngrams = get_ngrams(words, n)
            if ngrams:
                distinct_ratio = len(set(ngrams)) / len(ngrams)
                setattr(metrics, f"distinct_{n}", distinct_ratio)

        # Type-Token Ratio
        unique_words = len(set(words))
        total_words = len(words)
        metrics.type_token_ratio = unique_words / total_words if total_words > 0 else 0

        # Log TTR (corrigé pour la longueur)
        if total_words > 1:
            metrics.log_type_token_ratio = math.log(unique_words) / math.log(total_words)

        # Entropie lexicale (Shannon)
        word_counts = Counter(words)
        total = sum(word_counts.values())
        entropy = 0.0
        for count in word_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        metrics.lexical_entropy = entropy

        # Yule's K (mesure de richesse vocabulaire)
        # K = 10000 * (Σ(fi * vi) - N) / N²
        # où fi = fréquence, vi = nombre de mots avec cette fréquence
        freq_of_freq = Counter(word_counts.values())
        M1 = total_words
        M2 = sum(f * f * v for f, v in freq_of_freq.items())
        if M1 > 0 and M1 != M2:
            K = 10000 * (M2 - M1) / (M1 * M1)
            metrics.vocabulary_richness = K

        # Hapax legomena (mots apparaissant une seule fois)
        hapax = sum(1 for count in word_counts.values() if count == 1)
        metrics.hapax_legomena_ratio = hapax / unique_words if unique_words > 0 else 0

        return metrics

    def _compute_repetition(
        self,
        text: str,
        tokens: Optional[List] = None
    ) -> RepetitionMetrics:
        """Calcule les métriques de répétition."""
        metrics = RepetitionMetrics()

        if not text.strip():
            return metrics

        words = text.lower().split()

        if len(words) == 0:
            return metrics

        # Taux de répétition N-grams
        def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

        def repetition_rate(ngrams: List[Tuple[str, ...]]) -> float:
            if not ngrams:
                return 0.0
            counts = Counter(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            return repeated / len(ngrams)

        for n in [1, 2, 3, 4]:
            ngrams = get_ngrams(words, n)
            rate = repetition_rate(ngrams)
            setattr(metrics, f"rep_{n}", rate)

        # Plus longue chaîne de répétition consécutive
        max_repeat = 0
        current_repeat = 0
        prev_word = None
        for word in words:
            if word == prev_word:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
            prev_word = word
        metrics.max_consecutive_repeat = max_repeat

        # Détection de boucles
        metrics.loop_detected, metrics.loop_pattern = self._detect_loops(words)

        # Taux de répétition de mots consécutifs
        if len(words) > 1:
            consecutive_repeats = sum(
                1 for i in range(1, len(words)) if words[i] == words[i-1]
            )
            metrics.word_repetition_rate = consecutive_repeats / (len(words) - 1)

        # Répétition de phrases (groupes de 3+ mots)
        if len(words) >= 6:
            trigrams = get_ngrams(words, 3)
            trigram_counts = Counter(trigrams)
            repeated_phrases = sum(1 for c in trigram_counts.values() if c > 1)
            metrics.phrase_repetition_rate = repeated_phrases / len(trigrams) if trigrams else 0

        return metrics

    def _detect_loops(self, words: List[str]) -> Tuple[bool, Optional[str]]:
        """Détecte les boucles de répétition dans le texte."""
        if len(words) < 6:
            return False, None

        # Chercher des patterns répétés de longueur 2 à 10
        for pattern_len in range(2, min(11, len(words) // 2)):
            for start in range(len(words) - pattern_len * 2):
                pattern = words[start:start + pattern_len]

                # Compter combien de fois le pattern se répète consécutivement
                repeat_count = 1
                pos = start + pattern_len
                while pos + pattern_len <= len(words):
                    if words[pos:pos + pattern_len] == pattern:
                        repeat_count += 1
                        pos += pattern_len
                    else:
                        break

                # Si le pattern se répète 3+ fois, c'est une boucle
                if repeat_count >= 3:
                    return True, " ".join(pattern)

        return False, None

    def _compute_coherence(
        self,
        text: str,
        token_ids: Optional[torch.Tensor] = None
    ) -> CoherenceMetrics:
        """Calcule les métriques de cohérence et fluence."""
        metrics = CoherenceMetrics()

        if not text.strip():
            metrics.gibberish_score = 1.0
            return metrics

        # Détection de problèmes via patterns
        metrics.has_special_char_spam = bool(self.special_char_pattern.search(text))
        metrics.has_broken_words = bool(self.broken_word_pattern.search(text))

        # Détection de mélange de langues (caractères non-ASCII fréquents)
        non_ascii_matches = self.non_ascii_pattern.findall(text)
        non_ascii_ratio = sum(len(m) for m in non_ascii_matches) / len(text) if text else 0
        metrics.has_mixed_languages = non_ascii_ratio > 0.1

        # Score de charabia
        gibberish_indicators = [
            metrics.has_special_char_spam,
            metrics.has_broken_words,
            metrics.has_mixed_languages,
            bool(self.isolated_numbers_pattern.search(text)),
        ]

        # Analyse de la structure des mots
        words = text.split()
        if words:
            # Ratio de mots très longs (>15 caractères) ou très courts (<2)
            abnormal_words = sum(
                1 for w in words if len(w) > 15 or (len(w) < 2 and not w.isalpha())
            )
            abnormal_ratio = abnormal_words / len(words)

            # Ratio de mots avec chiffres mélangés
            alphanumeric_words = sum(
                1 for w in words if any(c.isdigit() for c in w) and any(c.isalpha() for c in w)
            )
            alphanumeric_ratio = alphanumeric_words / len(words)

            # Score de charabia
            metrics.gibberish_score = min(1.0, (
                sum(gibberish_indicators) * 0.15 +
                abnormal_ratio * 0.4 +
                alphanumeric_ratio * 0.3
            ))
        else:
            metrics.gibberish_score = 1.0

        # Score de cohérence (inverse du charabia, pondéré par diversité)
        metrics.coherence_score = max(0.0, 1.0 - metrics.gibberish_score)

        # Probabilités si model disponible
        if self.model and self.tokenizer:
            try:
                self.model.eval()
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

                if input_ids.size(1) > 1:
                    with torch.no_grad():
                        logits = self.model(input_ids)
                        probs = F.softmax(logits, dim=-1)

                        # Probabilités des tokens réels
                        shift_probs = probs[:, :-1, :]
                        shift_labels = input_ids[:, 1:]
                        token_probs = torch.gather(
                            shift_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                        ).squeeze(-1)

                        metrics.avg_token_probability = token_probs.mean().item()
                        metrics.min_token_probability = token_probs.min().item()
                        metrics.low_prob_token_ratio = (token_probs < 0.01).float().mean().item()
            except Exception:
                pass

        return metrics

    def _compute_overall_score(
        self,
        metrics: GenerationQualityMetrics
    ) -> Tuple[float, str]:
        """
        Calcule un score global (0-100) et une note.

        Pondération:
        - Cohérence: 30%
        - Diversité: 25%
        - Non-répétition: 25%
        - Perplexité (si disponible): 20%
        """
        score = 0.0

        # Cohérence (30%)
        coherence_score = metrics.coherence.coherence_score * 100
        score += coherence_score * 0.30

        # Diversité (25%) - basé sur distinct-2 et entropy
        diversity_score = (
            metrics.diversity.distinct_2 * 50 +
            min(metrics.diversity.lexical_entropy / 8, 1.0) * 50  # Normaliser entropy
        )
        score += diversity_score * 0.25

        # Non-répétition (25%)
        repetition_penalty = (
            metrics.repetition.rep_2 * 30 +
            metrics.repetition.rep_3 * 40 +
            (1.0 if metrics.repetition.loop_detected else 0.0) * 30
        )
        non_rep_score = max(0, 100 - repetition_penalty * 100)
        score += non_rep_score * 0.25

        # Perplexité (20%) - si disponible
        if metrics.perplexity.sequence_perplexity > 0:
            # PPL < 20 = bon, PPL > 500 = mauvais
            ppl = metrics.perplexity.sequence_perplexity
            if ppl < 20:
                ppl_score = 100
            elif ppl < 50:
                ppl_score = 80
            elif ppl < 100:
                ppl_score = 60
            elif ppl < 200:
                ppl_score = 40
            elif ppl < 500:
                ppl_score = 20
            else:
                ppl_score = 0
            score += ppl_score * 0.20
        else:
            # Sans perplexité, redistribuer le poids
            score = score / 0.80 * 1.0

        # Pénalités additionnelles
        if metrics.repetition.loop_detected:
            score *= 0.5
        if metrics.coherence.has_special_char_spam:
            score *= 0.7
        if metrics.coherence.gibberish_score > 0.5:
            score *= 0.6

        score = min(100, max(0, score))

        # Note
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 50:
            grade = "D"
        else:
            grade = "F"

        return score, grade

    def _generate_diagnostics(
        self,
        metrics: GenerationQualityMetrics
    ) -> Tuple[List[str], List[str]]:
        """Génère des diagnostics et recommandations."""
        issues = []
        recommendations = []

        # Problèmes de perplexité
        if metrics.perplexity.sequence_perplexity > 500:
            issues.append("Perplexité très élevée - modèle non convergé")
            recommendations.append("Continuer l'entraînement ou vérifier la qualité des données")
        elif metrics.perplexity.sequence_perplexity > 100:
            issues.append("Perplexité élevée - apprentissage incomplet")
            recommendations.append("Entraîner plus longtemps ou ajuster le learning rate")

        # Problèmes de diversité
        if metrics.diversity.distinct_2 < 0.3:
            issues.append("Faible diversité lexicale")
            recommendations.append("Augmenter la température ou réduire le repetition_penalty")

        # Problèmes de répétition
        if metrics.repetition.loop_detected:
            issues.append(f"Boucle de répétition détectée: '{metrics.repetition.loop_pattern}'")
            recommendations.append("Augmenter no_repeat_ngram_size ou repetition_penalty")
        elif metrics.repetition.rep_3 > 0.3:
            issues.append("Taux élevé de répétition de trigrams")
            recommendations.append("Ajuster les paramètres de sampling (top_k, top_p)")

        # Problèmes de cohérence
        if metrics.coherence.gibberish_score > 0.5:
            issues.append("Génération incohérente (charabia)")
            recommendations.append("Vérifier l'entraînement du modèle ou réduire la température")

        if metrics.coherence.has_broken_words:
            issues.append("Mots cassés ou mal formés détectés")
            recommendations.append("Vérifier le tokenizer et les special tokens")

        if metrics.coherence.has_special_char_spam:
            issues.append("Spam de caractères spéciaux")
            recommendations.append("Filtrer les caractères spéciaux ou ajuster le sampling")

        # Recommandations générales basées sur la note
        if metrics.quality_grade == "F":
            recommendations.append("⚠️ Qualité critique - revoir l'architecture ou les données d'entraînement")
        elif metrics.quality_grade == "D":
            recommendations.append("Qualité insuffisante - continuer l'entraînement et monitorer")

        return issues, recommendations

    def evaluate_batch(
        self,
        prompts: List[str],
        generated_texts: List[str],
        generation_times: Optional[List[float]] = None,
    ) -> List[GenerationQualityMetrics]:
        """
        Évalue un batch de générations.

        Args:
            prompts: Liste des prompts
            generated_texts: Liste des textes générés
            generation_times: Liste des temps de génération (optionnel)

        Returns:
            Liste de GenerationQualityMetrics
        """
        if generation_times is None:
            generation_times = [0.0] * len(prompts)

        results = []
        for prompt, gen_text, gen_time in zip(prompts, generated_texts, generation_times):
            metrics = self.evaluate(prompt, gen_text, gen_time)
            results.append(metrics)

        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques résumées sur l'historique des évaluations.

        Returns:
            Dict avec statistiques moyennes, min, max, etc.
        """
        if not self.evaluation_history:
            return {}

        n = len(self.evaluation_history)

        # Extraire les métriques
        scores = [m.overall_score for m in self.evaluation_history]
        ppls = [m.perplexity.sequence_perplexity for m in self.evaluation_history if m.perplexity.sequence_perplexity > 0]
        div_2 = [m.diversity.distinct_2 for m in self.evaluation_history]
        rep_2 = [m.repetition.rep_2 for m in self.evaluation_history]
        coherence = [m.coherence.coherence_score for m in self.evaluation_history]

        grades = Counter(m.quality_grade for m in self.evaluation_history)

        def safe_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            arr = np.array(values)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        return {
            "num_evaluations": n,
            "overall_score": safe_stats(scores),
            "perplexity": safe_stats(ppls) if ppls else None,
            "distinct_2": safe_stats(div_2),
            "repetition_2": safe_stats(rep_2),
            "coherence": safe_stats(coherence),
            "grade_distribution": dict(grades),
            "issues_frequency": self._count_issues(),
        }

    def _count_issues(self) -> Dict[str, int]:
        """Compte la fréquence des problèmes détectés."""
        issue_counts: Dict[str, int] = defaultdict(int)
        for m in self.evaluation_history:
            for issue in m.issues:
                # Simplifier l'issue pour comptage
                key = issue.split(":")[0].split("-")[0].strip()
                issue_counts[key] += 1
        return dict(issue_counts)

    def generate_report(
        self,
        output_path: Union[str, Path],
        format: str = "all",
    ) -> Dict[str, str]:
        """
        Génère des rapports détaillés.

        Args:
            output_path: Chemin de base pour les fichiers
            format: "json", "csv", "txt", ou "all"

        Returns:
            Dict avec les chemins des fichiers générés
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format in ("json", "all"):
            json_path = output_path.with_suffix(f".{timestamp}.json")
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_summary_statistics(),
                "evaluations": [m.to_dict() for m in self.evaluation_history],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            generated_files["json"] = str(json_path)

        if format in ("csv", "all"):
            csv_path = output_path.with_suffix(f".{timestamp}.csv")
            self._write_csv_report(csv_path)
            generated_files["csv"] = str(csv_path)

        if format in ("txt", "all"):
            txt_path = output_path.with_suffix(f".{timestamp}.txt")
            self._write_text_report(txt_path)
            generated_files["txt"] = str(txt_path)

        return generated_files

    def _write_csv_report(self, path: Path):
        """Écrit le rapport au format CSV."""
        import csv

        headers = [
            "timestamp", "prompt", "generated_text",
            "overall_score", "quality_grade",
            "ppl_sequence", "ppl_token",
            "distinct_1", "distinct_2", "distinct_3",
            "rep_1", "rep_2", "rep_3", "loop_detected",
            "coherence_score", "gibberish_score",
            "num_tokens", "tokens_per_second",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i, m in enumerate(self.evaluation_history):
                row = [
                    datetime.now().isoformat(),
                    m.prompt[:100],
                    m.generated_only[:200],
                    f"{m.overall_score:.2f}",
                    m.quality_grade,
                    f"{m.perplexity.sequence_perplexity:.2f}",
                    f"{m.perplexity.token_perplexity:.2f}",
                    f"{m.diversity.distinct_1:.4f}",
                    f"{m.diversity.distinct_2:.4f}",
                    f"{m.diversity.distinct_3:.4f}",
                    f"{m.repetition.rep_1:.4f}",
                    f"{m.repetition.rep_2:.4f}",
                    f"{m.repetition.rep_3:.4f}",
                    m.repetition.loop_detected,
                    f"{m.coherence.coherence_score:.4f}",
                    f"{m.coherence.gibberish_score:.4f}",
                    m.num_tokens,
                    f"{m.tokens_per_second:.2f}",
                ]
                writer.writerow(row)

    def _write_text_report(self, path: Path):
        """Écrit le rapport au format texte lisible."""
        summary = self.get_summary_statistics()

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT D'ÉVALUATION DE QUALITÉ DES GÉNÉRATIONS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("RÉSUMÉ STATISTIQUE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Nombre d'évaluations: {summary.get('num_evaluations', 0)}\n\n")

            if "overall_score" in summary:
                stats = summary["overall_score"]
                f.write(f"Score Global:\n")
                f.write(f"  Moyenne: {stats['mean']:.2f}/100\n")
                f.write(f"  Écart-type: {stats['std']:.2f}\n")
                f.write(f"  Min-Max: {stats['min']:.2f} - {stats['max']:.2f}\n\n")

            if summary.get("perplexity"):
                stats = summary["perplexity"]
                f.write(f"Perplexité:\n")
                f.write(f"  Moyenne: {stats['mean']:.2f}\n")
                f.write(f"  Min-Max: {stats['min']:.2f} - {stats['max']:.2f}\n\n")

            f.write(f"Distribution des Notes:\n")
            for grade, count in sorted(summary.get("grade_distribution", {}).items()):
                f.write(f"  {grade}: {count}\n")
            f.write("\n")

            f.write(f"Problèmes les plus fréquents:\n")
            for issue, count in sorted(
                summary.get("issues_frequency", {}).items(),
                key=lambda x: -x[1]
            )[:5]:
                f.write(f"  - {issue}: {count} occurrences\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("ÉVALUATIONS DÉTAILLÉES\n")
            f.write("=" * 80 + "\n\n")

            for i, m in enumerate(self.evaluation_history, 1):
                f.write(f"[{i}] Score: {m.overall_score:.1f}/100 ({m.quality_grade})\n")
                f.write(f"    Prompt: {m.prompt[:60]}{'...' if len(m.prompt) > 60 else ''}\n")
                f.write(f"    Généré: {m.generated_only[:80]}{'...' if len(m.generated_only) > 80 else ''}\n")
                f.write(f"    PPL: {m.perplexity.sequence_perplexity:.2f} | ")
                f.write(f"Distinct-2: {m.diversity.distinct_2:.3f} | ")
                f.write(f"Rep-2: {m.repetition.rep_2:.3f}\n")
                if m.issues:
                    f.write(f"    Issues: {', '.join(m.issues[:2])}\n")
                f.write("\n")

    def log_to_tensorboard(
        self,
        writer,
        step: int,
        metrics: Optional[GenerationQualityMetrics] = None,
        prefix: str = "generation",
    ):
        """
        Log les métriques vers TensorBoard.

        Args:
            writer: SummaryWriter de TensorBoard
            step: Step actuel
            metrics: Métriques à logger (ou dernière évaluation)
            prefix: Préfixe pour les noms de métriques
        """
        if metrics is None:
            if not self.evaluation_history:
                return
            metrics = self.evaluation_history[-1]

        # Score global
        writer.add_scalar(f"{prefix}/overall_score", metrics.overall_score, step)

        # Perplexité
        if metrics.perplexity.sequence_perplexity > 0:
            writer.add_scalar(f"{prefix}/perplexity/sequence", metrics.perplexity.sequence_perplexity, step)
            writer.add_scalar(f"{prefix}/perplexity/token_avg", metrics.perplexity.token_perplexity, step)
            writer.add_scalar(f"{prefix}/perplexity/confident_ratio", metrics.perplexity.confident_tokens_ratio, step)

        # Diversité
        writer.add_scalar(f"{prefix}/diversity/distinct_1", metrics.diversity.distinct_1, step)
        writer.add_scalar(f"{prefix}/diversity/distinct_2", metrics.diversity.distinct_2, step)
        writer.add_scalar(f"{prefix}/diversity/distinct_3", metrics.diversity.distinct_3, step)
        writer.add_scalar(f"{prefix}/diversity/entropy", metrics.diversity.lexical_entropy, step)
        writer.add_scalar(f"{prefix}/diversity/ttr", metrics.diversity.type_token_ratio, step)

        # Répétition
        writer.add_scalar(f"{prefix}/repetition/rep_1", metrics.repetition.rep_1, step)
        writer.add_scalar(f"{prefix}/repetition/rep_2", metrics.repetition.rep_2, step)
        writer.add_scalar(f"{prefix}/repetition/rep_3", metrics.repetition.rep_3, step)
        writer.add_scalar(f"{prefix}/repetition/loop_detected", float(metrics.repetition.loop_detected), step)

        # Cohérence
        writer.add_scalar(f"{prefix}/coherence/score", metrics.coherence.coherence_score, step)
        writer.add_scalar(f"{prefix}/coherence/gibberish", metrics.coherence.gibberish_score, step)

        # Performance
        if metrics.tokens_per_second > 0:
            writer.add_scalar(f"{prefix}/performance/tokens_per_second", metrics.tokens_per_second, step)


# ======================================================================
# Standalone Evaluation Functions
# ======================================================================

def evaluate_text_quality(text: str) -> Dict[str, Any]:
    """
    Évalue rapidement la qualité d'un texte sans modèle.

    Args:
        text: Texte à évaluer

    Returns:
        Dict avec métriques de diversité, répétition, cohérence
    """
    evaluator = GenerationEvaluator()
    metrics = evaluator.evaluate("", text, compute_perplexity=False)
    return metrics.to_dict()


def compare_generations(
    texts: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare plusieurs générations.

    Args:
        texts: Liste de textes à comparer
        labels: Labels optionnels pour chaque texte

    Returns:
        Dict avec métriques comparatives
    """
    evaluator = GenerationEvaluator()

    if labels is None:
        labels = [f"text_{i}" for i in range(len(texts))]

    results = {}
    for label, text in zip(labels, texts):
        metrics = evaluator.evaluate("", text, compute_perplexity=False)
        results[label] = {
            "score": metrics.overall_score,
            "grade": metrics.quality_grade,
            "distinct_2": metrics.diversity.distinct_2,
            "rep_2": metrics.repetition.rep_2,
            "coherence": metrics.coherence.coherence_score,
        }

    # Classement
    ranked = sorted(results.items(), key=lambda x: -x[1]["score"])
    results["_ranking"] = [label for label, _ in ranked]

    return results


__all__ = [
    "GenerationEvaluator",
    "GenerationQualityMetrics",
    "PerplexityMetrics",
    "DiversityMetrics",
    "RepetitionMetrics",
    "CoherenceMetrics",
    "evaluate_text_quality",
    "compare_generations",
]
