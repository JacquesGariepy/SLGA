#!/usr/bin/env python3
"""
💬 Chat Interactif avec le Reasoning Model

Interface en ligne de commande pour tester le modèle.

Usage:
    python chat_reasoning.py --checkpoint checkpoints/reasoning/checkpoint_best.pt
    python chat_reasoning.py --checkpoint ckpt.pt --temperature 0.7

Commandes spéciales:
    /quit, /exit  - Quitter
    /clear        - Effacer l'historique
    /config       - Afficher la config
    /tot          - Activer Tree of Thought
    /cot          - Revenir à Chain of Thought
    /temp 0.5     - Changer la température
"""

import os
import sys
import argparse
import torch
from typing import Optional

from train_reasoning_simple import (
    ReasoningModel,
    TrainConfig,
    SimpleTokenizer,
    SPECIAL_TOKENS,
)


class ReasoningChat:
    """Interface de chat interactive."""

    def __init__(
        self,
        model: ReasoningModel,
        tokenizer: SimpleTokenizer,
        device: str = "cuda",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_tot = False
        self.history = []

    def generate_response(self, question: str) -> str:
        """Génère une réponse avec raisonnement."""
        # Formater le prompt
        prompt = f"Question: {question}\n<think>"
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)

        # Générer
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        # Décoder
        full_text = self.tokenizer.decode(output_ids[0].tolist())

        # Extraire la partie générée
        generated = full_text[len(prompt):]

        return generated

    def format_response(self, response: str) -> str:
        """Formate la réponse pour l'affichage."""
        # Coloriser les tokens spéciaux
        formatted = response
        formatted = formatted.replace('<think>', '\n🤔 \033[94m<think>\033[0m')
        formatted = formatted.replace('</think>', '\033[94m</think>\033[0m')
        formatted = formatted.replace('<step>', '\n  📍 \033[93m')
        formatted = formatted.replace('<answer>', '\n\n✅ \033[92m<answer>\033[0m ')
        formatted = formatted.replace('</answer>', ' \033[92m</answer>\033[0m')

        return formatted

    def handle_command(self, cmd: str) -> Optional[str]:
        """Gère les commandes spéciales."""
        parts = cmd.strip().split()
        command = parts[0].lower()

        if command in ['/quit', '/exit', '/q']:
            return "EXIT"

        elif command == '/clear':
            self.history = []
            return "Historique effacé."

        elif command == '/config':
            return f"""Configuration:
  Temperature: {self.temperature}
  Max tokens: {self.max_tokens}
  Mode: {'Tree of Thought' if self.use_tot else 'Chain of Thought'}
  Device: {self.device}"""

        elif command == '/tot':
            self.use_tot = True
            return "Mode Tree of Thought activé (non implémenté dans cette version)."

        elif command == '/cot':
            self.use_tot = False
            return "Mode Chain of Thought activé."

        elif command == '/temp' and len(parts) > 1:
            try:
                self.temperature = float(parts[1])
                return f"Température: {self.temperature}"
            except:
                return "Usage: /temp 0.7"

        elif command == '/help':
            return """Commandes disponibles:
  /quit, /exit  - Quitter
  /clear        - Effacer l'historique
  /config       - Afficher la configuration
  /tot          - Activer Tree of Thought
  /cot          - Activer Chain of Thought
  /temp <val>   - Changer la température
  /help         - Afficher cette aide"""

        return None

    def run(self):
        """Boucle principale du chat."""
        print("\n" + "=" * 60)
        print("🧠 SLGA Reasoning Model - Chat Interactif")
        print("=" * 60)
        print("Tapez votre question ou /help pour l'aide")
        print("=" * 60 + "\n")

        while True:
            try:
                # Input
                user_input = input("\033[96m❓ Vous:\033[0m ").strip()

                if not user_input:
                    continue

                # Commande?
                if user_input.startswith('/'):
                    result = self.handle_command(user_input)
                    if result == "EXIT":
                        print("\n👋 Au revoir!")
                        break
                    if result:
                        print(f"\033[90m{result}\033[0m\n")
                    continue

                # Générer la réponse
                print("\033[90m⏳ Réflexion en cours...\033[0m")

                response = self.generate_response(user_input)
                formatted = self.format_response(response)

                print(f"\n🤖 Modèle:{formatted}\n")

                # Historique
                self.history.append({
                    "question": user_input,
                    "response": response,
                })

            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break
            except Exception as e:
                print(f"\033[91m❌ Erreur: {e}\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with Reasoning Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger le modèle
    print(f"📂 Chargement: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt.get("config", TrainConfig())
    if isinstance(config, dict):
        config = TrainConfig(**config)

    model = ReasoningModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"✓ Modèle chargé ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Tokenizer
    tokenizer = SimpleTokenizer()

    # Chat
    chat = ReasoningChat(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    chat.run()


if __name__ == "__main__":
    main()
