import pytest
from simple_tokenizer import simple_tokenizer  # adjust import to your file


@pytest.fixture
def tokenizer():
    texts = [
        "hello world",
        "machine learning is fun"
    ]
    tok = simple_tokenizer()
    tok.build_vocab(texts)
    return tok


def test_special_tokens(tokenizer):
    # Special tokens exist in the vocab
    assert "<PAD>" in tokenizer.word_to_id
    assert "<UNK>" in tokenizer.word_to_id
    assert "<BOS>" in tokenizer.word_to_id
    assert "<EOS>" in tokenizer.word_to_id


def test_vocab_building(tokenizer):
    # Words from training set should exist in vocab
    for word in ["hello", "world", "machine", "learning", "is", "fun"]:
        assert word in tokenizer.word_to_id
    # Vocab size should be at least number of words + 4 special tokens
    assert tokenizer.vocab_size >= 10


def test_encode_known_words(tokenizer):
    # Encode a sentence with known words only
    ids = tokenizer.encode("hello world")
    # The encoded IDs should correspond to the words
    assert ids == [tokenizer.word_to_id["hello"], tokenizer.word_to_id["world"]]


def test_encode_unknown_word(tokenizer):
    # Encode a sentence with an unknown word
    ids = tokenizer.encode("hello unknownword")
    unk_id = tokenizer.word_to_id["<UNK>"]
    # <UNK> should appear for the unknown word
    assert ids == [tokenizer.word_to_id["hello"], unk_id]


def test_decode(tokenizer):
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    # Decoded text should match the original words
    assert decoded_text == text


def test_round_trip(tokenizer):
    original_text = "machine learning"
    ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(ids)
    # Decoding after encoding should preserve words
    assert decoded_text == original_text


def test_unknown_decode(tokenizer):
    # Decoding an unknown ID should return <UNK>
    unknown_id = 999
    decoded = tokenizer.decode([unknown_id])
    assert decoded == "<UNK>"
