"""
Module d'affichage en temps réel des métriques d'entraînement.

Features:
- Affichage console coloré et formaté
- Barres de progression visuelles
- Graphiques ASCII pour tendances
- Calcul automatique des best values
- Détection d'anomalies (NaN, instabilités)
- ETA et vitesse de training
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Deque, Dict, List


# Couleurs ANSI
class Colors:
    """Codes couleur ANSI pour terminal"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Couleurs de base
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Couleurs brillantes
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class Symbols:
    """Symboles Unicode pour affichage"""

    ARROW_UP = "↑"
    ARROW_DOWN = "↓"
    ARROW_RIGHT = "→"
    CHECK = "✓"
    CROSS = "✗"
    STAR = "★"
    WARNING = "⚠"
    FIRE = "🔥"
    ROCKET = "🚀"
    CLOCK = "⏱"
    CHART = "📊"
    GPU = "🖥"
    MEMORY = "💾"
    BRAIN = "🧠"
    TARGET = "🎯"
    PROGRESS = "▶"
    BLOCK_FULL = "█"
    BLOCK_PARTIAL = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


@dataclass
class MetricsHistory:
    """Historique des métriques pour calcul de tendances"""

    maxlen: int = 50

    # Loss et PPL
    loss: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    ppl: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    val_loss: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    val_ppl: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # Performance
    throughput: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    gpu_mem: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    # Training
    grad_norm: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    lr: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def add(self, key: str, value: float) -> None:
        """Ajoute une valeur à l'historique"""
        if hasattr(self, key):
            getattr(self, key).append(value)

    def get_trend(self, key: str, window: int = 10) -> str:
        """Calcule la tendance (↑↓→) sur les dernières valeurs"""
        if not hasattr(self, key):
            return Symbols.ARROW_RIGHT

        data = list(getattr(self, key))
        if len(data) < 2:
            return Symbols.ARROW_RIGHT

        # Prendre les 'window' dernières valeurs
        recent = data[-min(window, len(data)) :]

        if len(recent) < 2:
            return Symbols.ARROW_RIGHT

        # Calculer tendance linéaire simple
        first_half = sum(recent[: len(recent) // 2]) / max(1, len(recent) // 2)
        second_half = sum(recent[len(recent) // 2 :]) / max(1, len(recent) - len(recent) // 2)

        diff = second_half - first_half
        threshold = abs(first_half) * 0.05  # 5% de changement

        if abs(diff) < threshold:
            return Symbols.ARROW_RIGHT
        if diff > 0:
            return Symbols.ARROW_UP
        return Symbols.ARROW_DOWN


class LiveMetricsDisplay:
    """
    Affichage en temps réel des métriques d'entraînement.

    Usage:
        display = LiveMetricsDisplay(max_steps=100000)
        display.update(step=1000, loss=2.5, ppl=12.2, ...)
    """

    def __init__(
        self,
        max_steps: int,
        log_every: int = 50,
        width: int = 100,
        compact: bool = False,
    ) -> None:
        self.max_steps: int = max_steps
        self.log_every: int = log_every
        self.width: int = width
        self.compact: bool = compact

        # Tracking
        self.history: MetricsHistory = MetricsHistory()
        self.start_time: float = time.time()
        self.last_log_time: float = time.time()
        self.steps_since_last: int = 0

        # Best values
        self.best_loss: float = float("inf")
        self.best_val_loss: float = float("inf")
        self.best_ppl: float = float("inf")

        # Anomaly tracking
        self.nan_count: int = 0
        self.instability_count: int = 0

    def format_number(self, value: float, precision: int = 4, width: int = 8) -> str:
        """Formate un nombre avec couleur selon magnitude"""
        if math.isnan(value) or math.isinf(value):
            return f"{Colors.BRIGHT_RED}{value!s:>{width}}{Colors.RESET}"

        # Format selon magnitude
        if abs(value) < 0.001:
            formatted = f"{value:.2e}"
        elif abs(value) < 1:
            formatted = f"{value:.{precision}f}"
        elif abs(value) < 100:
            formatted = f"{value:.{max(0, precision-1)}f}"
        else:
            formatted = f"{value:.0f}"

        return f"{formatted:>{width}}"

    def format_percentage(self, value: float, width: int = 6) -> str:
        """Formate un pourcentage avec couleur"""
        pct = value * 100
        if pct < 25:
            color = Colors.RED
        elif pct < 50:
            color = Colors.YELLOW
        elif pct < 75:
            color = Colors.GREEN
        else:
            color = Colors.BRIGHT_GREEN

        return f"{color}{pct:>{width-1}.1f}%{Colors.RESET}"

    def make_progress_bar(self, progress: float, width: int = 30) -> str:
        """Crée une barre de progression visuelle"""
        filled = int(progress * width)
        empty = width - filled

        # Couleur selon progression
        if progress < 0.33:
            color = Colors.RED
        elif progress < 0.66:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN

        bar = f"{color}{Symbols.BLOCK_FULL * filled}{Colors.DIM}{'─' * empty}{Colors.RESET}"
        return f"[{bar}] {self.format_percentage(progress, width=6)}"

    def make_mini_chart(self, values: List[float], height: int = 3, width: int = 20) -> List[str]:
        """Crée un mini graphique ASCII"""
        if not values or len(values) < 2:
            return [" " * width for _ in range(height)]

        # Normaliser valeurs
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val == 0:
            range_val = 1

        # Échantillonner pour width
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values + [values[-1]] * (width - len(values))

        # Créer graphique
        lines = []
        for row in range(height):
            threshold = min_val + (height - row - 0.5) * range_val / height
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += Symbols.BLOCK_FULL
                else:
                    line += " "
            lines.append(line)

        return lines

    def format_time(self, seconds: float) -> str:
        """Formate une durée en heures:minutes:secondes"""
        if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
            return "??:??:??"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def detect_anomalies(self, metrics: Dict[str, float]) -> List[str]:
        """Détecte les anomalies dans les métriques"""
        warnings = []

        # NaN detection
        if math.isnan(metrics.get("loss", 0)):
            self.nan_count += 1
            warnings.append(f"{Symbols.WARNING} NaN detected in loss ({self.nan_count}x)")

        # Loss explosion
        if metrics.get("loss", 0) > 10:
            warnings.append(f"{Symbols.WARNING} Loss explosion: {metrics['loss']:.2f}")

        # Gradient explosion
        if metrics.get("grad_norm", 0) > 10:
            warnings.append(f"{Symbols.WARNING} Gradient explosion: {metrics['grad_norm']:.2f}")

        # GPU memory warning
        if metrics.get("gpu_memory_used_pct", 0) > 0.95:
            warnings.append(
                f"{Symbols.WARNING} GPU memory critical: {metrics['gpu_memory_used_pct']*100:.0f}%"
            )

        # Throughput drop
        if len(self.history.throughput) > 5:
            recent_avg = sum(list(self.history.throughput)[-5:]) / 5
            if metrics.get("tokens_per_sec", 0) < recent_avg * 0.5:
                warnings.append(f"{Symbols.WARNING} Throughput dropped 50%")

        return warnings

    def update(self, **metrics: float) -> None:
        """
        Met à jour l'affichage avec nouvelles métriques.

        Args:
            step: Step actuel
            loss: Loss d'entraînement
            ppl: Perplexité
            lr: Learning rate
            grad_norm: Norme du gradient
            seq_len: Longueur de séquence
            global_weight: Poids attention globale
            tokens_per_sec: Tokens/sec
            gpu_memory_gb: Mémoire GPU utilisée (GB)
            gpu_memory_total_gb: Mémoire GPU totale (GB)
            val_loss: (optionnel) Loss de validation
            val_ppl: (optionnel) PPL de validation
            num_landmarks: (optionnel) Nombre de landmarks
            spacing_loss: (optionnel) Loss de spacing
            sparsity_loss: (optionnel) Loss de sparsité
        """
        step: int = int(metrics.get("step", 0))

        # Skip si pas au bon interval
        if step % self.log_every != 0 and step != 0:
            return

        # Extraire métriques
        loss: float = float(metrics.get("loss", 0.0))
        ppl = metrics.get("ppl", 0.0)
        lr = metrics.get("lr", 0.0)
        grad_norm = metrics.get("grad_norm", 0.0)

        # Ajouter à l'historique
        self.history.add("loss", loss)
        self.history.add("ppl", ppl)
        self.history.add("lr", lr)
        self.history.add("grad_norm", grad_norm)

        if "val_loss" in metrics:
            self.history.add("val_loss", metrics["val_loss"])
            self.history.add("val_ppl", metrics.get("val_ppl", math.exp(metrics["val_loss"])))

        # Throughput
        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        if tokens_per_sec > 0:
            self.history.add("throughput", tokens_per_sec)

        # GPU memory
        gpu_mem = metrics.get("gpu_memory_gb", 0)
        if gpu_mem > 0:
            self.history.add("gpu_mem", gpu_mem)

        # Update best values
        if loss < self.best_loss and loss > 0:
            self.best_loss = loss
        if ppl < self.best_ppl and ppl > 0:
            self.best_ppl = ppl
        if "val_loss" in metrics and metrics["val_loss"] < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]

        # Calculate performance
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_since_last = current_time - self.last_log_time

        steps_done = step
        steps_remaining = self.max_steps - step
        progress = step / self.max_steps

        # ETA
        if steps_done > 0:
            time_per_step = elapsed_total / steps_done
            eta_seconds = time_per_step * steps_remaining
        else:
            eta_seconds = 0

        # Detect anomalies
        warnings = self.detect_anomalies(metrics)

        # Build display
        self._render(
            step=step,
            progress=progress,
            metrics=metrics,
            elapsed_total=elapsed_total,
            eta_seconds=eta_seconds,
            warnings=warnings,
        )

        # Update tracking
        self.last_log_time = current_time
        self.steps_since_last = 0

    def _render(
        self,
        step: int,
        progress: float,
        metrics: Dict[str, float],
        elapsed_total: float,
        eta_seconds: float,
        warnings: List[str],
    ) -> None:
        """Rendu de l'affichage console"""

        # Clear previous (optionnel, peut causer flickering)
        # print("\033[2J\033[H", end="")

        # Separator
        print()
        print(Colors.BOLD + "=" * self.width + Colors.RESET)

        # === HEADER ===
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{Symbols.ROCKET} SLGA Training Live Metrics{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}Step {step:,} / {self.max_steps:,} ({progress*100:.1f}%) │ Elapsed: {self.format_time(elapsed_total)} │ ETA: {self.format_time(eta_seconds)}{Colors.RESET}"
        )

        # Progress bar
        print(self.make_progress_bar(progress, width=self.width - 20))
        print()

        # === MAIN METRICS ===
        print(f"{Colors.BOLD}📊 Core Metrics{Colors.RESET}")
        print(f"{'─' * self.width}")

        loss = metrics.get("loss", 0.0)
        ppl = metrics.get("ppl", 0.0)
        lr = metrics.get("lr", 0.0)

        loss_trend = self.history.get_trend("loss")
        ppl_trend = self.history.get_trend("ppl")

        # Loss avec couleur
        if loss > 5:
            loss_color = Colors.RED
        elif loss > 3:
            loss_color = Colors.YELLOW
        else:
            loss_color = Colors.GREEN

        print(
            f"  Loss:       {loss_color}{self.format_number(loss, precision=4)}{Colors.RESET} {loss_trend}  │  Best: {Colors.GREEN}{self.best_loss:.4f}{Colors.RESET}"
        )
        print(
            f"  Perplexity: {loss_color}{self.format_number(ppl, precision=2)}{Colors.RESET} {ppl_trend}  │  Best: {Colors.GREEN}{self.best_ppl:.2f}{Colors.RESET}"
        )
        # Format lr safely (handle None)
        lr_str = f"{lr:.6e}" if lr is not None else "N/A"
        print(
            f"  Learn Rate: {Colors.CYAN}{lr_str}{Colors.RESET}  │  Grad Norm: {Colors.CYAN}{metrics.get('grad_norm', 0.0):.4f}{Colors.RESET}"
        )
        print()

        # === VALIDATION (si disponible) ===
        if "val_loss" in metrics:
            print(f"{Colors.BOLD}{Symbols.TARGET} Validation{Colors.RESET}")
            print(f"{'─' * self.width}")

            val_loss = metrics["val_loss"]
            val_ppl = metrics.get("val_ppl", math.exp(val_loss))
            gap = val_loss - loss

            gap_color = Colors.GREEN if gap < 0.5 else Colors.YELLOW if gap < 1.5 else Colors.RED

            print(
                f"  Val Loss:   {self.format_number(val_loss, precision=4)}  │  Val PPL: {self.format_number(val_ppl, precision=2)}"
            )
            print(
                f"  Train/Val Gap: {gap_color}{gap:+.4f}{Colors.RESET}  │  Best Val: {Colors.GREEN}{self.best_val_loss:.4f}{Colors.RESET}"
            )
            print()

        # === PERFORMANCE ===
        print(f"{Colors.BOLD}{Symbols.ROCKET} Performance{Colors.RESET}")
        print(f"{'─' * self.width}")

        tokens_per_sec = metrics.get("tokens_per_sec", 0)
        seq_len = metrics.get("seq_len", 0)
        samples_per_sec = tokens_per_sec / max(1, seq_len)

        print(
            f"  Throughput:  {Colors.BRIGHT_GREEN}{tokens_per_sec:>8,.0f}{Colors.RESET} tok/s  │  {Colors.GREEN}{samples_per_sec:>6,.1f}{Colors.RESET} samples/s"
        )
        print(f"  Seq Length:  {Colors.CYAN}{seq_len:>8,}{Colors.RESET} tokens")

        # GPU
        gpu_mem_gb = metrics.get("gpu_memory_gb", 0)
        gpu_mem_total_gb = metrics.get("gpu_memory_total_gb", 24)
        gpu_mem_pct = gpu_mem_gb / gpu_mem_total_gb if gpu_mem_total_gb > 0 else 0

        if gpu_mem_pct > 0.9:
            mem_color = Colors.RED
        elif gpu_mem_pct > 0.75:
            mem_color = Colors.YELLOW
        else:
            mem_color = Colors.GREEN

        print(
            f"  GPU Memory:  {mem_color}{gpu_mem_gb:>6.2f}{Colors.RESET} / {gpu_mem_total_gb:.1f} GB  ({self.format_percentage(gpu_mem_pct)})"
        )
        print()

        # === LANDMARKS (si activés) ===
        if "num_landmarks" in metrics and metrics["num_landmarks"] > 0:
            print(f"{Colors.BOLD}{Symbols.BRAIN} Landmarks{Colors.RESET}")
            print(f"{'─' * self.width}")

            num_lm = metrics["num_landmarks"]
            spacing_loss = metrics.get("spacing_loss", 0.0)
            sparsity_loss = metrics.get("sparsity_loss", 0.0)
            global_weight = metrics.get("global_weight", 1.0)

            print(
                f"  Selected:     {Colors.CYAN}{num_lm:>4}{Colors.RESET}  │  Global Weight: {self.format_percentage(global_weight)}"
            )
            print(
                f"  Spacing Loss: {self.format_number(spacing_loss, precision=4)}  │  Sparsity Loss: {self.format_number(sparsity_loss, precision=4)}"
            )
            print()

        # === MINI CHART (tendance loss) ===
        if len(self.history.loss) > 10 and not self.compact:
            print(
                f"{Colors.BOLD}{Symbols.CHART} Loss Trend (last {len(self.history.loss)} steps){Colors.RESET}"
            )
            print(f"{'─' * self.width}")

            chart_lines = self.make_mini_chart(
                list(self.history.loss), height=3, width=min(50, self.width - 10)
            )
            for line in chart_lines:
                print(f"  {Colors.DIM}{line}{Colors.RESET}")
            print()

        # === WARNINGS ===
        if warnings:
            print(f"{Colors.BOLD}{Colors.BRIGHT_RED}{Symbols.WARNING} WARNINGS{Colors.RESET}")
            print(f"{'─' * self.width}")
            for warning in warnings:
                print(f"  {Colors.BRIGHT_YELLOW}{warning}{Colors.RESET}")
            print()

        # Footer
        print(Colors.BOLD + "=" * self.width + Colors.RESET)
        print()


def test_display() -> None:
    """Test du display avec données simulées"""
    import random

    display = LiveMetricsDisplay(max_steps=100000, log_every=1, width=100)

    print("Testing LiveMetricsDisplay...")
    print("Simulating training steps...\n")

    for step in range(0, 1000, 50):
        # Simulate decreasing loss
        base_loss = 5.0 * math.exp(-step / 500)
        loss = base_loss + random.uniform(-0.1, 0.1)
        ppl = math.exp(loss)

        metrics = {
            "step": step,
            "loss": loss,
            "ppl": ppl,
            "lr": 2e-4 * (1 - step / 1000),
            "grad_norm": 1.0 + random.uniform(-0.2, 0.2),
            "seq_len": 512 + (step // 100) * 128,
            "global_weight": min(1.0, step / 500),
            "tokens_per_sec": 4000 + random.uniform(-500, 500),
            "gpu_memory_gb": 12.5 + random.uniform(-0.5, 0.5),
            "gpu_memory_total_gb": 24.0,
            "num_landmarks": 24,
            "spacing_loss": 0.01 + random.uniform(-0.002, 0.002),
            "sparsity_loss": 0.005 + random.uniform(-0.001, 0.001),
        }

        # Add validation every 500 steps
        if step % 500 == 0 and step > 0:
            metrics["val_loss"] = loss + random.uniform(0.5, 1.5)
            metrics["val_ppl"] = math.exp(metrics["val_loss"])

        display.update(**metrics)
        time.sleep(0.5)  # Simulate time between steps


if __name__ == "__main__":
    test_display()
