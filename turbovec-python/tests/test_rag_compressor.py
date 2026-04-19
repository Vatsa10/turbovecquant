from __future__ import annotations

from turbovec.llm import RAGCompressor

from _llm_fakes import FakeEmbedder


def test_compress_respects_token_budget():
    rag = RAGCompressor(
        embedder=FakeEmbedder(),
        chunk_size=40,
        chunk_overlap=0,
        token_counter=lambda s: len(s.split()),
    )
    docs = [
        "the capital of france is paris and it is a beautiful european city",
        "the theory of relativity was developed by albert einstein in the early twentieth century",
        "bananas grow in tropical climates and are yellow when ripe",
    ]
    rag.add_documents(docs)
    packed = rag.compress("where is paris", token_budget=10)
    assert isinstance(packed, str)
    assert len(packed.split()) <= 10 + 2  # separator fudge


def test_compress_returns_highest_scoring_first():
    rag = RAGCompressor(
        embedder=FakeEmbedder(),
        chunk_size=40,
        chunk_overlap=0,
        token_counter=lambda s: len(s.split()),
    )
    rag.add_documents(
        [
            "paris capital france europe",
            "bananas tropical yellow fruit",
            "einstein relativity physics",
        ]
    )
    chunks = rag.compress("paris france", token_budget=200, return_chunks=True)
    assert chunks
    assert "paris" in chunks[0].text


def test_empty_corpus():
    rag = RAGCompressor(embedder=FakeEmbedder())
    assert rag.compress("anything", token_budget=100) == ""
    assert rag.compress("anything", token_budget=100, return_chunks=True) == []
