"""
Affichage en temps réel pour training avec mise à jour à chaque step.

Features:
- Mise à jour ligne par ligne (pas de clear screen)
- Barre de progression qui se met à jour chaque step
- Métriques live: loss, PPL, throughput, GPU, ETA
- Utilise \r pour réecrire la même ligne
"""

from __future__ import annotations

from collections import deque
import math
import sys
import time
from typing import Deque, Optional


class Colors:
    """Codes couleur ANSI"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


class RealtimeTrainingDisplay:
    """
    Affichage en temps réel du training qui se met à jour à CHAQUE step.

    Features:
    - Ligne de progression mise à jour en temps réel
    - Métriques clés visibles en permanence
    - Logs détaillés tous les N steps
    - Pas de flickering (utilise \r au lieu de clear)
    """

    def __init__(
        self,
        max_steps: int,
        detail_every: int = 50,
        width: int = 100,
    ) -> None:
        self.max_steps: int = max_steps
        self.detail_every: int = detail_every
        self.width: int = width

        # Tracking
        self.start_time: float = time.time()
        self.last_step_time: float = time.time()
        self.step_times: Deque[float] = deque(maxlen=20)  # Pour smooth throughput

        # Best values
        self.best_loss: float = float("inf")
        self.best_ppl: float = float("inf")

        # Current values (pour affichage continu)
        self.current_step: int = 0
        self.current_loss: float = 0.0
        self.current_ppl: float = 0.0
        self.current_lr: float = 0.0
        self.current_throughput: float = 0.0
        self.current_gpu_mem: float = 0.0
        self.current_gpu_total: float = 24.0
        self.current_seq_len: int = 0
        self.current_global_weight: float = 0.0
        self.current_num_landmarks: int = 0
        self.current_active_landmarks: int = 0

        # Header affiché une seule fois
        self._print_header()

    def _print_header(self) -> None:
        """Affiche le header une seule fois au début"""
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'='*self.width}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}🚀 SLGA Training - Real-Time Display{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{'='*self.width}{Colors.RESET}")
        print()

    def _format_time(self, seconds: float) -> str:
        """Formate durée en HH:MM:SS"""
        if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
            return "??:??:??"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _make_progress_bar(self, progress: float, width: int = 40) -> str:
        """Crée barre de progression compacte"""
        filled = int(progress * width)
        empty = width - filled

        # Couleur selon progression
        if progress < 0.33:
            color = Colors.RED
        elif progress < 0.66:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN

        bar = f"{color}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.RESET}"
        return f"[{bar}]"

    def _color_for_value(self, value: float, good_low: bool = True) -> str:
        """Retourne couleur selon valeur (rouge=mauvais, vert=bon)"""
        if math.isnan(value) or math.isinf(value):
            return Colors.BRIGHT_RED

        # Pour loss/ppl: bas=bon
        if good_low:
            if value > 5:
                return Colors.RED
            if value > 3:
                return Colors.YELLOW
            return Colors.GREEN
        # Pour throughput: haut=bon
        if value < 2000:
            return Colors.RED
        if value < 4000:
            return Colors.YELLOW
        return Colors.GREEN

    def update_live(
        self,
        step: int,
        loss: Optional[float] = None,
        ppl: Optional[float] = None,
        lr: Optional[float] = None,
        throughput: Optional[float] = None,
        gpu_mem_gb: Optional[float] = None,
        gpu_total_gb: Optional[float] = None,
        seq_len: Optional[int] = None,
        global_weight: Optional[float] = None,
        num_landmarks: Optional[int] = None,
        active_landmarks: Optional[int] = None,
    ) -> None:
        """
        Mise à jour live (appelée à CHAQUE step).
        Réécrit la ligne de progression en temps réel.

        Args:
            step: Step actuel
            loss: Loss (optionnel, garde dernière valeur si None)
            ppl: Perplexité
            lr: Learning rate
            throughput: Tokens/sec
            gpu_mem_gb: Mémoire GPU
            gpu_total_gb: Total GPU
            seq_len: Sequence length
            global_weight: Global attention weight
            num_landmarks: Number of total landmarks
            active_landmarks: Number of active landmarks
        """
        # Mettre à jour valeurs courantes
        self.current_step = step
        if loss is not None:
            self.current_loss = loss
            if loss < self.best_loss and loss > 0:
                self.best_loss = loss
        if ppl is not None:
            self.current_ppl = ppl
            if ppl < self.best_ppl and ppl > 0:
                self.best_ppl = ppl
        if lr is not None:
            self.current_lr = lr
        if throughput is not None:
            self.current_throughput = throughput
        if gpu_mem_gb is not None:
            self.current_gpu_mem = gpu_mem_gb
        if gpu_total_gb is not None:
            self.current_gpu_total = gpu_total_gb
        if seq_len is not None:
            self.current_seq_len = seq_len
        else:
            self.current_seq_len = getattr(self, "current_seq_len", 0)
        if global_weight is not None:
            self.current_global_weight = global_weight
        else:
            self.current_global_weight = getattr(self, "current_global_weight", 0.0)
        if num_landmarks is not None:
            self.current_num_landmarks = num_landmarks
        else:
            self.current_num_landmarks = getattr(self, "current_num_landmarks", 0)
        if active_landmarks is not None:
            self.current_active_landmarks = active_landmarks
        else:
            self.current_active_landmarks = getattr(self, "current_active_landmarks", 0)

        # Calcul timing
        current_time: float = time.time()
        elapsed_total: float = current_time - self.start_time
        step_time: float = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time

        # ETA
        eta_seconds: float
        if step > 0:
            avg_step_time: float = sum(self.step_times) / len(self.step_times)
            remaining_steps: int = self.max_steps - step
            eta_seconds = avg_step_time * remaining_steps
        else:
            eta_seconds = 0.0

        progress: float = step / self.max_steps

        # GPU percentage
        gpu_pct: float = (
            self.current_gpu_mem / self.current_gpu_total if self.current_gpu_total > 0 else 0.0
        )

        # Couleurs
        loss_color = self._color_for_value(self.current_loss, good_low=True)
        throughput_color = self._color_for_value(self.current_throughput, good_low=False)

        if gpu_pct > 0.9:
            gpu_color = Colors.RED
        elif gpu_pct > 0.75:
            gpu_color = Colors.YELLOW
        else:
            gpu_color = Colors.GREEN

        # Construire ligne de progression (réécrite en place)
        progress_bar = self._make_progress_bar(progress, width=30)

        # Landmarks display
        if self.current_num_landmarks > 0:
            lm_display = f"{self.current_num_landmarks}→{self.current_active_landmarks}"
        else:
            lm_display = "N/A"

        line = (
            f"\r{Colors.BOLD}Step {step:>6}/{self.max_steps}{Colors.RESET} "
            f"{progress_bar} {progress*100:>5.1f}% │ "
            f"⏱ {self._format_time(elapsed_total)} │ "
            f"ETA {self._format_time(eta_seconds)} │ "
            f"Loss {loss_color}{self.current_loss:6.3f}{Colors.RESET} │ "
            f"PPL {loss_color}{self.current_ppl:>7.1f}{Colors.RESET} │ "
            f"LR {Colors.CYAN}{self.current_lr:.1e}{Colors.RESET} │ "
            f"SeqLen {self.current_seq_len:>4d} │ "
            f"GW {self.current_global_weight:.2f} │ "
            f"LM {lm_display:>7s} │ "
            f"{throughput_color}{self.current_throughput:>5.0f}{Colors.RESET} tok/s │ "
            f"GPU {gpu_color}{gpu_pct*100:>4.1f}%{Colors.RESET}"
        )

        # Écrire ligne (flush pour update immédiat)
        sys.stdout.write(line)
        sys.stdout.flush()

    def print_detailed(
        self,
        step: int,
        loss: float,
        ppl: float,
        lr: float,
        grad_norm: float,
        seq_len: int,
        global_weight: float,
        throughput: float,
        gpu_mem_gb: float,
        gpu_total_gb: float,
        num_landmarks: int = 0,
        spacing_loss: float = 0.0,
        sparsity_loss: float = 0.0,
    ) -> None:
        """
        Affichage détaillé (tous les detail_every steps).
        Saute une ligne puis affiche les détails, puis reprend ligne live.
        """

        # Nouvelle ligne (pour ne pas écraser la ligne live)
        def _fmt_metric(value: float) -> str:
            """Formate une métrique avec précision adaptative."""
            if abs(value) < 1e-4 and value != 0.0:
                return f"{value:.3e}"
            return f"{value:.4f}"

        print()
        print()

        # Séparateur
        print(f"{Colors.BOLD}{Colors.DIM}{'─'*self.width}{Colors.RESET}")

        # Core metrics
        loss_color = self._color_for_value(loss, good_low=True)

        print(
            f"{Colors.BOLD}📊 Step {step:,}{Colors.RESET} │ "
            f"Loss: {loss_color}{loss:.4f}{Colors.RESET} (best: {Colors.GREEN}{self.best_loss:.4f}{Colors.RESET}) │ "
            f"PPL: {loss_color}{ppl:.2f}{Colors.RESET} (best: {Colors.GREEN}{self.best_ppl:.2f}{Colors.RESET})"
        )

        # Training params (handle None values)
        lr_str = f"{lr:.6e}" if lr is not None else "N/A"
        grad_str = f"{grad_norm:.4f}" if grad_norm is not None else "N/A"
        seq_str = f"{seq_len:,}" if seq_len is not None else "N/A"
        gw_str = f"{global_weight:.2f}" if global_weight is not None else "N/A"

        print(
            f"   LR: {Colors.CYAN}{lr_str}{Colors.RESET} │ "
            f"Grad: {Colors.CYAN}{grad_str}{Colors.RESET} │ "
            f"SeqLen: {Colors.CYAN}{seq_str}{Colors.RESET} │ "
            f"GW: {Colors.CYAN}{gw_str}{Colors.RESET}"
        )

        # Performance (handle None values)
        throughput = throughput or 0
        gpu_mem_gb = gpu_mem_gb or 0
        gpu_total_gb = gpu_total_gb or 1  # Avoid division by zero

        throughput_color = self._color_for_value(throughput, good_low=False)
        gpu_pct = gpu_mem_gb / gpu_total_gb if gpu_total_gb > 0 else 0

        if gpu_pct > 0.9:
            gpu_color = Colors.RED
        elif gpu_pct > 0.75:
            gpu_color = Colors.YELLOW
        else:
            gpu_color = Colors.GREEN

        print(
            f"   Throughput: {throughput_color}{throughput:,.0f}{Colors.RESET} tok/s │ "
            f"GPU: {gpu_color}{gpu_mem_gb:.2f}{Colors.RESET}/{gpu_total_gb:.1f} GB "
            f"({gpu_color}{gpu_pct*100:.1f}%{Colors.RESET})"
        )

        # Landmarks si activés
        if num_landmarks > 0 or spacing_loss > 0.0 or sparsity_loss > 0.0:
            print(
                f"   Landmarks: {Colors.MAGENTA}{num_landmarks}{Colors.RESET} │ "
                f"Spacing: {Colors.MAGENTA}{_fmt_metric(spacing_loss)}{Colors.RESET} │ "
                f"Sparsity: {Colors.MAGENTA}{_fmt_metric(sparsity_loss)}{Colors.RESET}"
            )

        print(f"{Colors.BOLD}{Colors.DIM}{'─'*self.width}{Colors.RESET}")
        print()

    def print_validation(self, step: int, val_loss: float, val_ppl: float) -> None:
        """Affichage validation"""
        print()
        print()
        print(f"{Colors.BOLD}{Colors.BRIGHT_GREEN}🎯 VALIDATION @ Step {step:,}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.DIM}{'─'*self.width}{Colors.RESET}")

        gap = val_loss - self.current_loss
        gap_color = Colors.GREEN if gap < 0.5 else Colors.YELLOW if gap < 1.5 else Colors.RED

        print(
            f"   Val Loss: {Colors.CYAN}{val_loss:.4f}{Colors.RESET} │ "
            f"Val PPL: {Colors.CYAN}{val_ppl:.2f}{Colors.RESET} │ "
            f"Gap: {gap_color}{gap:+.4f}{Colors.RESET}"
        )

        print(f"{Colors.BOLD}{Colors.DIM}{'─'*self.width}{Colors.RESET}")
        print()

    def print_warning(self, message: str) -> None:
        """Affiche un warning"""
        print()
        print(f"{Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")
        print()


def test_realtime() -> None:
    """Test du display en temps réel"""
    import random

    display = RealtimeTrainingDisplay(max_steps=1000, detail_every=10)

    print("Testing real-time display...")
    print("Watch the live progress bar update every step!")
    print()

    for step in range(1, 101):
        # Simulate decreasing loss
        loss = 5.0 * math.exp(-step / 50) + random.uniform(-0.1, 0.1)
        ppl = math.exp(loss)
        lr = 2e-4 * min(1.0, step / 10)
        throughput = 4000 + random.uniform(-500, 500)
        gpu_mem = 12.5 + random.uniform(-0.5, 0.5)

        # Update live (chaque step)
        display.update_live(
            step=step,
            loss=loss,
            ppl=ppl,
            lr=lr,
            throughput=throughput,
            gpu_mem_gb=gpu_mem,
            gpu_total_gb=24.0,
        )

        # Detailed print tous les 10 steps
        if step % 10 == 0:
            display.print_detailed(
                step=step,
                loss=loss,
                ppl=ppl,
                lr=lr,
                grad_norm=1.0 + random.uniform(-0.2, 0.2),
                seq_len=384 + (step // 10) * 64,
                global_weight=min(1.0, step / 50),
                throughput=throughput,
                gpu_mem_gb=gpu_mem,
                gpu_total_gb=24.0,
                num_landmarks=24,
                spacing_loss=0.01 + random.uniform(-0.002, 0.002),
                sparsity_loss=0.005 + random.uniform(-0.001, 0.001),
            )

        # Simulate validation
        if step % 50 == 0:
            display.print_validation(
                step=step,
                val_loss=loss + random.uniform(0.5, 1.5),
                val_ppl=math.exp(loss + random.uniform(0.5, 1.5)),
            )

        # Simulate step time (variable)
        time.sleep(random.uniform(0.05, 0.15))

    # Final newline
    print()
    print()
    print(f"{Colors.BRIGHT_GREEN}✓ Test completed!{Colors.RESET}")


if __name__ == "__main__":
    test_realtime()
