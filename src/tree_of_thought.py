# tree_of_thought.py
"""
Tree of Thought (ToT) pour SLGA-Reasoning

Implémente une exploration arborescente du raisonnement:
- Génère plusieurs branches à chaque étape
- Évalue et élague les branches prometteuses
- Permet le backtracking
- Trouve le meilleur chemin de raisonnement

Basé sur: "Tree of Thoughts: Deliberate Problem Solving with LLMs" (Yao et al., 2023)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq


class SearchStrategy(Enum):
    """Stratégies de recherche dans l'arbre."""
    BFS = "bfs"           # Breadth-First Search
    DFS = "dfs"           # Depth-First Search
    BEAM = "beam"         # Beam Search
    MCTS = "mcts"         # Monte Carlo Tree Search
    BEST_FIRST = "best_first"  # Best-First Search (A*)


@dataclass
class ThoughtNode:
    """
    Noeud dans l'arbre de pensée.

    Représente une étape de raisonnement avec:
    - Le contenu de la pensée
    - Son score d'évaluation
    - Ses enfants (branches alternatives)
    - Son parent (pour backtracking)
    """
    thought: str                          # Contenu de cette étape
    hidden_state: Optional[torch.Tensor]  # État caché du modèle
    score: float = 0.0                    # Score d'évaluation
    depth: int = 0                        # Profondeur dans l'arbre
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    is_terminal: bool = False             # True si c'est une réponse finale
    visits: int = 0                       # Pour MCTS
    value_sum: float = 0.0                # Pour MCTS

    @property
    def value(self) -> float:
        """Valeur moyenne (pour MCTS)."""
        return self.value_sum / max(1, self.visits)

    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score (pour MCTS)."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value
        exploration = math.sqrt(2 * math.log(self.parent.visits + 1) / self.visits)
        return exploitation + exploration

    def get_path(self) -> List['ThoughtNode']:
        """Retourne le chemin depuis la racine."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_full_reasoning(self) -> str:
        """Retourne le raisonnement complet depuis la racine."""
        path = self.get_path()
        return " → ".join([n.thought for n in path if n.thought])


@dataclass
class ToTConfig:
    """Configuration pour Tree of Thought."""
    # Structure de l'arbre
    max_depth: int = 8                    # Profondeur maximale
    branching_factor: int = 3             # Branches par noeud
    min_branches: int = 2                 # Minimum de branches à explorer

    # Recherche
    search_strategy: SearchStrategy = SearchStrategy.BEAM
    beam_width: int = 5                   # Pour beam search
    max_iterations: int = 100             # Limite d'itérations

    # Évaluation
    use_value_network: bool = True        # Utiliser un réseau de valeur
    use_self_evaluation: bool = True      # Le modèle s'auto-évalue
    pruning_threshold: float = 0.3        # Seuil pour élaguer

    # MCTS spécifique
    mcts_simulations: int = 50            # Simulations par noeud
    mcts_temperature: float = 1.0         # Température d'exploration

    # Génération
    thought_temperature: float = 0.7      # Température pour générer les pensées
    thought_max_tokens: int = 100         # Tokens max par pensée


class ThoughtGenerator(nn.Module):
    """
    Génère des pensées alternatives à chaque étape.

    Utilise le modèle de base avec différents sampling
    pour produire des branches diverses.
    """

    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Projecteur de diversité: encourage des pensées différentes
        self.diversity_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Tête de type de pensée (déduction, hypothèse, vérification, etc.)
        self.thought_type_head = nn.Linear(embed_dim, 5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_branches: int = 3,
        existing_thoughts: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Génère des représentations pour plusieurs pensées alternatives.

        Args:
            hidden_states: (B, L, D) état actuel
            num_branches: nombre de branches à générer
            existing_thoughts: pensées existantes (pour diversité)

        Returns:
            branch_states: (B, num_branches, D) états pour chaque branche
            thought_types: (B, num_branches, 5) types de pensée
        """
        B, L, D = hidden_states.shape

        # Pooling sur la séquence
        pooled = hidden_states.mean(dim=1)  # (B, D)

        # Générer des états diversifiés
        branch_states = []
        for i in range(num_branches):
            # Ajouter du bruit pour diversité
            noise = torch.randn_like(pooled) * (0.1 * (i + 1))
            diverse_state = self.diversity_proj(pooled + noise)

            # Répulsion des pensées existantes
            if existing_thoughts:
                for existing in existing_thoughts:
                    sim = F.cosine_similarity(diverse_state, existing, dim=-1)
                    # Pousser loin des pensées similaires
                    diverse_state = diverse_state - 0.1 * sim.unsqueeze(-1) * existing

            branch_states.append(diverse_state)

        branch_states = torch.stack(branch_states, dim=1)  # (B, num_branches, D)

        # Types de pensée
        thought_types = self.thought_type_head(branch_states)  # (B, num_branches, 5)

        return branch_states, thought_types


class ThoughtEvaluator(nn.Module):
    """
    Évalue la qualité d'une pensée/étape de raisonnement.

    Critères d'évaluation:
    1. Cohérence avec le contexte
    2. Progrès vers la solution
    3. Validité logique
    4. Originalité (pas de répétition)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Encodeur de pensée
        self.thought_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encodeur de contexte (question + pensées précédentes)
        self.context_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Têtes d'évaluation
        self.coherence_head = nn.Linear(hidden_dim * 2, 1)
        self.progress_head = nn.Linear(hidden_dim * 2, 1)
        self.validity_head = nn.Linear(hidden_dim * 2, 1)
        self.final_score_head = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        thought_state: torch.Tensor,
        context_state: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Évalue une pensée dans son contexte.

        Args:
            thought_state: (B, D) état de la pensée à évaluer
            context_state: (B, D) état du contexte (question + historique)
            return_components: si True, retourne les scores individuels

        Returns:
            dict avec score final et optionnellement les composants
        """
        # Encoder
        thought_enc = self.thought_encoder(thought_state)  # (B, hidden)
        context_enc = self.context_encoder(context_state)  # (B, hidden)

        # Concatener
        combined = torch.cat([thought_enc, context_enc], dim=-1)  # (B, 2*hidden)

        # Scores individuels
        coherence = torch.sigmoid(self.coherence_head(combined))
        progress = torch.sigmoid(self.progress_head(combined))
        validity = torch.sigmoid(self.validity_head(combined))

        # Score final (combinaison apprise)
        final_score = torch.sigmoid(self.final_score_head(combined))

        result = {"score": final_score.squeeze(-1)}

        if return_components:
            result["coherence"] = coherence.squeeze(-1)
            result["progress"] = progress.squeeze(-1)
            result["validity"] = validity.squeeze(-1)

        return result


class TreeOfThought(nn.Module):
    """
    Module principal Tree of Thought.

    Orchestre:
    1. La génération de pensées alternatives
    2. L'évaluation des branches
    3. La recherche dans l'arbre
    4. La sélection du meilleur chemin
    """

    def __init__(
        self,
        model: nn.Module,  # Le modèle SLGA-Reasoning
        config: ToTConfig,
        embed_dim: int = 768,
        vocab_size: int = 50261,
    ):
        super().__init__()

        self.model = model
        self.config = config
        self.embed_dim = embed_dim

        # Composants
        self.thought_generator = ThoughtGenerator(embed_dim, vocab_size)
        self.thought_evaluator = ThoughtEvaluator(embed_dim)

        # Value network (prédit la valeur d'un état)
        if config.use_value_network:
            self.value_network = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Tanh(),  # Valeur entre -1 et 1
            )

    def generate_thoughts(
        self,
        node: ThoughtNode,
        input_ids: torch.Tensor,
        num_thoughts: int = 3,
    ) -> List[ThoughtNode]:
        """
        Génère des pensées alternatives à partir d'un noeud.

        Args:
            node: Noeud parent
            input_ids: Tokens du contexte actuel
            num_thoughts: Nombre de pensées à générer

        Returns:
            Liste de noeuds enfants
        """
        device = input_ids.device

        # Forward pour obtenir les états cachés
        with torch.no_grad():
            outputs = self.model(input_ids, return_reasoning_info=True)
            hidden_states = outputs.get("hidden_states", outputs["logits"])

        # Si hidden_states est logits, on fait un pooling simple
        if hidden_states.dim() == 3 and hidden_states.size(-1) == self.model.cfg.vocab_size:
            # C'est logits, on utilise le dernier état avant lm_head
            context_state = hidden_states.mean(dim=1)
        else:
            context_state = hidden_states[:, -1, :]  # Dernier état

        # Générer des états diversifiés
        existing_states = [c.hidden_state for c in node.children if c.hidden_state is not None]
        branch_states, thought_types = self.thought_generator(
            hidden_states if hidden_states.dim() == 3 else hidden_states.unsqueeze(1),
            num_branches=num_thoughts,
            existing_thoughts=existing_states,
        )

        children = []
        for i in range(num_thoughts):
            branch_state = branch_states[:, i, :]  # (B, D)

            # Générer le texte de la pensée
            thought_text = self._generate_thought_text(
                input_ids,
                branch_state,
                thought_type=thought_types[:, i, :].argmax(dim=-1).item(),
            )

            # Évaluer la pensée
            eval_result = self.thought_evaluator(
                branch_state,
                context_state,
                return_components=True,
            )

            # Créer le noeud enfant
            child = ThoughtNode(
                thought=thought_text,
                hidden_state=branch_state.detach(),
                score=eval_result["score"].item(),
                depth=node.depth + 1,
                parent=node,
                is_terminal=self._is_terminal(thought_text),
            )

            children.append(child)
            node.children.append(child)

        return children

    def _generate_thought_text(
        self,
        input_ids: torch.Tensor,
        branch_state: torch.Tensor,
        thought_type: int = 0,
    ) -> str:
        """Génère le texte d'une pensée."""
        # Types de pensée
        type_prompts = [
            "Let me deduce: ",      # Déduction
            "I hypothesize: ",      # Hypothèse
            "Checking: ",           # Vérification
            "Breaking down: ",      # Décomposition
            "Alternatively: ",      # Alternative
        ]

        prefix = type_prompts[thought_type % len(type_prompts)]

        # Génération simplifiée (en pratique, utiliserait le modèle)
        # TODO: Implémenter génération complète avec le modèle
        return f"{prefix}[Generated thought at depth {branch_state.mean().item():.2f}]"

    def _is_terminal(self, thought: str) -> bool:
        """Vérifie si une pensée est terminale (réponse finale)."""
        terminal_indicators = ["answer:", "therefore:", "conclusion:", "final answer:"]
        thought_lower = thought.lower()
        return any(ind in thought_lower for ind in terminal_indicators)

    def search(
        self,
        input_ids: torch.Tensor,
        question: str = "",
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """
        Recherche le meilleur chemin de raisonnement.

        Args:
            input_ids: Tokens de la question
            question: Texte de la question (optionnel)

        Returns:
            best_node: Meilleur noeud terminal trouvé
            all_nodes: Tous les noeuds explorés
        """
        strategy = self.config.search_strategy

        if strategy == SearchStrategy.BEAM:
            return self._beam_search(input_ids, question)
        elif strategy == SearchStrategy.BFS:
            return self._bfs_search(input_ids, question)
        elif strategy == SearchStrategy.DFS:
            return self._dfs_search(input_ids, question)
        elif strategy == SearchStrategy.MCTS:
            return self._mcts_search(input_ids, question)
        elif strategy == SearchStrategy.BEST_FIRST:
            return self._best_first_search(input_ids, question)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        question: str,
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """Beam Search: garde les K meilleures branches à chaque niveau."""

        # Racine
        root = ThoughtNode(thought=question, hidden_state=None, depth=0)
        beam = [root]
        all_nodes = [root]
        terminals = []

        for depth in range(self.config.max_depth):
            candidates = []

            for node in beam:
                if node.is_terminal:
                    terminals.append(node)
                    continue

                # Générer des enfants
                children = self.generate_thoughts(
                    node, input_ids,
                    num_thoughts=self.config.branching_factor
                )
                candidates.extend(children)
                all_nodes.extend(children)

            if not candidates:
                break

            # Garder les K meilleurs
            candidates.sort(key=lambda n: n.score, reverse=True)
            beam = candidates[:self.config.beam_width]

            # Élaguer les branches faibles
            beam = [n for n in beam if n.score >= self.config.pruning_threshold]

        # Ajouter les noeuds terminaux restants
        terminals.extend([n for n in beam if n.is_terminal])

        # Retourner le meilleur
        if terminals:
            best = max(terminals, key=lambda n: n.score)
        else:
            best = max(all_nodes, key=lambda n: n.score)

        return best, all_nodes

    def _bfs_search(
        self,
        input_ids: torch.Tensor,
        question: str,
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """Breadth-First Search: explore niveau par niveau."""
        from collections import deque

        root = ThoughtNode(thought=question, hidden_state=None, depth=0)
        queue = deque([root])
        all_nodes = [root]
        best_terminal = None

        iterations = 0
        while queue and iterations < self.config.max_iterations:
            node = queue.popleft()
            iterations += 1

            if node.is_terminal:
                if best_terminal is None or node.score > best_terminal.score:
                    best_terminal = node
                continue

            if node.depth >= self.config.max_depth:
                continue

            children = self.generate_thoughts(node, input_ids)
            for child in children:
                if child.score >= self.config.pruning_threshold:
                    queue.append(child)
                    all_nodes.append(child)

        return best_terminal or root, all_nodes

    def _dfs_search(
        self,
        input_ids: torch.Tensor,
        question: str,
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """Depth-First Search avec backtracking."""

        root = ThoughtNode(thought=question, hidden_state=None, depth=0)
        all_nodes = [root]
        best_terminal = None

        def dfs(node: ThoughtNode, depth: int):
            nonlocal best_terminal

            if node.is_terminal:
                if best_terminal is None or node.score > best_terminal.score:
                    best_terminal = node
                return

            if depth >= self.config.max_depth:
                return

            children = self.generate_thoughts(node, input_ids)
            # Trier par score (explorer les meilleurs d'abord)
            children.sort(key=lambda n: n.score, reverse=True)

            for child in children:
                if child.score >= self.config.pruning_threshold:
                    all_nodes.append(child)
                    dfs(child, depth + 1)

                    # Backtrack si on a trouvé une solution
                    if best_terminal and best_terminal.score > 0.9:
                        return

        dfs(root, 0)
        return best_terminal or root, all_nodes

    def _mcts_search(
        self,
        input_ids: torch.Tensor,
        question: str,
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """Monte Carlo Tree Search."""

        root = ThoughtNode(thought=question, hidden_state=None, depth=0)
        all_nodes = [root]

        for _ in range(self.config.mcts_simulations):
            # 1. Selection: descendre jusqu'à une feuille
            node = root
            while node.children and not node.is_terminal:
                # UCB selection
                node = max(node.children, key=lambda n: n.ucb_score)

            # 2. Expansion: ajouter des enfants
            if not node.is_terminal and node.depth < self.config.max_depth:
                children = self.generate_thoughts(node, input_ids, num_thoughts=1)
                if children:
                    node = children[0]
                    all_nodes.append(node)

            # 3. Simulation: évaluer
            value = self._simulate(node, input_ids)

            # 4. Backpropagation
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        # Sélectionner le meilleur enfant de la racine
        if root.children:
            best = max(root.children, key=lambda n: n.visits)
            # Descendre jusqu'à un terminal
            while best.children:
                best = max(best.children, key=lambda n: n.visits)
        else:
            best = root

        return best, all_nodes

    def _simulate(self, node: ThoughtNode, input_ids: torch.Tensor) -> float:
        """Simulation rapide pour MCTS."""
        if node.is_terminal:
            return node.score

        # Simulation simple: utiliser le value network
        if self.config.use_value_network and node.hidden_state is not None:
            value = self.value_network(node.hidden_state)
            return value.item()

        return node.score

    def _best_first_search(
        self,
        input_ids: torch.Tensor,
        question: str,
    ) -> Tuple[ThoughtNode, List[ThoughtNode]]:
        """Best-First Search (A*-like)."""

        root = ThoughtNode(thought=question, hidden_state=None, depth=0)
        # Priority queue: (-score, node) car heapq est min-heap
        frontier = [(-root.score, id(root), root)]
        all_nodes = [root]
        best_terminal = None

        iterations = 0
        while frontier and iterations < self.config.max_iterations:
            _, _, node = heapq.heappop(frontier)
            iterations += 1

            if node.is_terminal:
                if best_terminal is None or node.score > best_terminal.score:
                    best_terminal = node
                continue

            if node.depth >= self.config.max_depth:
                continue

            children = self.generate_thoughts(node, input_ids)
            for child in children:
                if child.score >= self.config.pruning_threshold:
                    heapq.heappush(frontier, (-child.score, id(child), child))
                    all_nodes.append(child)

        return best_terminal or root, all_nodes

    def get_best_reasoning_path(
        self,
        input_ids: torch.Tensor,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Interface principale: trouve le meilleur chemin de raisonnement.

        Returns:
            dict avec:
                - path: Liste des pensées dans l'ordre
                - answer: Réponse finale
                - score: Score du chemin
                - tree_stats: Statistiques de l'arbre
        """
        best_node, all_nodes = self.search(input_ids, question)

        path = best_node.get_path()

        return {
            "path": [n.thought for n in path],
            "full_reasoning": best_node.get_full_reasoning(),
            "answer": path[-1].thought if path else "",
            "score": best_node.score,
            "depth": best_node.depth,
            "tree_stats": {
                "total_nodes": len(all_nodes),
                "max_depth_reached": max(n.depth for n in all_nodes),
                "terminals_found": sum(1 for n in all_nodes if n.is_terminal),
                "avg_branching": sum(len(n.children) for n in all_nodes) / max(1, len(all_nodes)),
            }
        }


def create_tree_of_thought(
    model: nn.Module,
    strategy: str = "beam",
    beam_width: int = 5,
    max_depth: int = 8,
    **kwargs
) -> TreeOfThought:
    """
    Factory pour créer un module Tree of Thought.

    Args:
        model: Modèle SLGA-Reasoning
        strategy: "beam", "bfs", "dfs", "mcts", "best_first"
        beam_width: Largeur du beam search
        max_depth: Profondeur maximale de l'arbre

    Returns:
        TreeOfThought configuré
    """
    strategy_map = {
        "beam": SearchStrategy.BEAM,
        "bfs": SearchStrategy.BFS,
        "dfs": SearchStrategy.DFS,
        "mcts": SearchStrategy.MCTS,
        "best_first": SearchStrategy.BEST_FIRST,
    }

    config = ToTConfig(
        search_strategy=strategy_map.get(strategy, SearchStrategy.BEAM),
        beam_width=beam_width,
        max_depth=max_depth,
        **kwargs
    )

    return TreeOfThought(
        model=model,
        config=config,
        embed_dim=getattr(model.cfg, 'embed_dim', 768),
        vocab_size=getattr(model.cfg, 'vocab_size', 50261),
    )


__all__ = [
    "SearchStrategy",
    "ThoughtNode",
    "ToTConfig",
    "ThoughtGenerator",
    "ThoughtEvaluator",
    "TreeOfThought",
    "create_tree_of_thought",
]
