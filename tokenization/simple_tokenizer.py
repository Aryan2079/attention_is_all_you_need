import numpy as np
from typing import List, Dict

class simple_tokenizer:
    """
    A word-level tokenizer with special tokens
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[str, int] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.word_to_id = {
            self.pad_token : 0,
            self.unk_token : 1,
            self.bos_token : 2,
            self.eos_token : 3
        }
 
        self.id_to_word = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.bos_token,
            3: self.eos_token
        }   
        
        self.vocab_size = 4

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts
        """

        unique_words = set()

        for text in texts:
            words = text.split()
            unique_words.update(words)

        for word in unique_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text : str) -> List[int]:
        """
        Convert text into list of token ids
        """

        text_list = text.split()
        ids = []

        # ids.append(self.word_to_id.get(self.bos_token))

        for word in text_list:
            if word not in self.word_to_id:
                ids.append(self.word_to_id.get(self.unk_token))
            else:
                ids.append(self.word_to_id.get(word))
        
        # ids.append(self.word_to_id.get(self.eos_token))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of ids back to text
        """
        words = []
        for word_id in ids:
            word = self.id_to_word.get(word_id, self.unk_token)
            words.append(word)
        
        return " ".join(words)