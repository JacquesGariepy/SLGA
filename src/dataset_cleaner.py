"""
Dataset wrapper pour nettoyer corruption Unicode et caractères problématiques.

Usage:
    ds_train = load_text_dataset(...)
    ds_train_clean = CleanedDataset(ds_train)
"""

from typing import Dict, Any
import re


class CleanedDataset:
    """
    Wrapper autour d'un dataset HuggingFace qui nettoie les textes.

    Nettoie:
    - Caractères Unicode de remplacement (�, U+FFFD)
    - Espaces multiples
    - Caractères de contrôle
    """

    def __init__(self, dataset, text_key: str = "text"):
        """
        Args:
            dataset: Dataset HuggingFace à wrapper
            text_key: Clé contenant le texte à nettoyer
        """
        self.dataset = dataset
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Retourne l'exemple nettoyé"""
        example = self.dataset[idx]

        # Si l'exemple est un dict avec text_key
        if isinstance(example, dict) and self.text_key in example:
            text = example[self.text_key]
            cleaned_text = self.clean_text(text)
            example_clean = example.copy()
            example_clean[self.text_key] = cleaned_text
            return example_clean

        # Si l'exemple est directement une string
        elif isinstance(example, str):
            return self.clean_text(example)

        # Sinon, retourner tel quel
        else:
            return example

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Nettoie un texte de ses caractères problématiques.

        Args:
            text: Texte à nettoyer

        Returns:
            text_clean: Texte nettoyé
        """
        if not isinstance(text, str):
            return text

        # 1. Remplacer caractères de remplacement Unicode (�)
        # U+FFFD (�) est inséré par Python quand l'encodage échoue
        text = text.replace('\ufffd', ' ')  # Unicode escape
        text = text.replace('�', ' ')       # Littéral

        # 2. Supprimer caractères de contrôle (sauf \n, \t)
        # Garde: espace (32), tab (9), newline (10), carriage return (13)
        # Supprime: tous les autres < 32 et DEL (127)
        text = ''.join(
            c for c in text
            if ord(c) >= 32 or c in '\n\t\r'
        )

        # 3. Normaliser espaces multiples (mais garder newlines)
        # Remplacer 3+ espaces consécutifs par 2 espaces
        text = re.sub(r' {3,}', '  ', text)

        # 4. Normaliser tabs multiples
        text = re.sub(r'\t+', '\t', text)

        # 5. Normaliser newlines multiples (max 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 6. Supprimer espaces en début/fin de lignes
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        return text

    def select(self, indices):
        """Support pour .select() comme HF datasets"""
        return CleanedDataset(
            self.dataset.select(indices),
            text_key=self.text_key
        )

    @property
    def column_names(self):
        """Expose les colonnes du dataset sous-jacent"""
        if hasattr(self.dataset, 'column_names'):
            return self.dataset.column_names
        return None

    @property
    def features(self):
        """Expose les features du dataset sous-jacent"""
        if hasattr(self.dataset, 'features'):
            return self.dataset.features
        return None


def test_cleaner():
    """Test unitaire du nettoyeur"""

    # Test 1: Caractères de remplacement
    text1 = "Hello � world � test"
    expected1 = "Hello  world  test"  # 3+ espaces → 2 espaces (1 espace + � → 2 espaces)
    result1 = CleanedDataset.clean_text(text1)
    assert result1 == expected1, f"Failed: Unicode replacement. Got '{result1}', expected '{expected1}'"

    # Test 2: Espaces multiples
    text2 = "Hello    world     test"
    expected2 = "Hello  world  test"
    assert CleanedDataset.clean_text(text2) == expected2, "Failed: Multiple spaces"

    # Test 3: Newlines multiples
    text3 = "Line1\n\n\n\nLine2"
    expected3 = "Line1\n\nLine2"
    assert CleanedDataset.clean_text(text3) == expected3, "Failed: Multiple newlines"

    # Test 4: Caractères de contrôle
    text4 = "Hello\x00\x01\x02world"
    expected4 = "Helloworld"
    assert CleanedDataset.clean_text(text4) == expected4, "Failed: Control chars"

    # Test 5: Mix complexe
    text5 = "  The capital  \x00  of\ufffd France   \n\n\n\n is    Paris  "
    expected5 = "The capital  of  France\n\nis  Paris"
    result5 = CleanedDataset.clean_text(text5)
    assert result5 == expected5, f"Failed: Complex mix. Got '{result5}', expected '{expected5}'"

    print("✅ All tests passed!")


if __name__ == "__main__":
    test_cleaner()
