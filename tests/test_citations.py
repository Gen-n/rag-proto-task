from rag.generate import AnswerGenerator


def test_normalize_grouped_citations():
    g = AnswerGenerator()
    text = "Answer uses two sources [Source 1, Source 2]."
    out = g._normalize_grouped_citations(text)
    assert out == "Answer uses two sources [Source 1][Source 2]."


def test_strip_invalid_sources_removes_out_of_range():
    g = AnswerGenerator()
    text = "Ok [Source 1] bad [Source 3]."
    out = g._strip_invalid_sources(text, max_source=2)
    assert out == "Ok [Source 1] bad ."


def test_strip_invalid_sources_keeps_in_range():
    g = AnswerGenerator()
    text = "A [Source 1] B [Source 2]."
    out = g._strip_invalid_sources(text, max_source=2)
    assert out == text


def test_extract_citations_single_source():
    g = AnswerGenerator()
    retrieved_docs = [
        {
            "content": "Encoder has N=6 layers.",
            "metadata": {
                "source": "Attention_is_all_you_need.pdf",
                "page": 3,
                "chunk_id": "p3:c0",
                "chunk_index": 0,
            },
        }
    ]

    answer = "Encoder uses N = 6 layers [Source 1]."
    citations = g._extract_citations(answer, retrieved_docs)

    assert len(citations) == 1
    c = citations[0]
    assert c["source_number"] == 1
    assert c["source"] == "Attention_is_all_you_need.pdf"
    assert c["page"] == 3
    assert c["chunk_id"] == "p3:c0"
    assert c["chunk_index"] == 0


def test_extract_citations_multiple_sources_dedup_and_sorted():
    g = AnswerGenerator()
    retrieved_docs = [
        {
            "content": "A",
            "metadata": {
                "source": "doc.pdf",
                "page": 1,
                "chunk_id": "c0",
                "chunk_index": 0,
            },
        },
        {
            "content": "B",
            "metadata": {
                "source": "doc.pdf",
                "page": 2,
                "chunk_id": "c1",
                "chunk_index": 1,
            },
        },
    ]

    answer = "Two cites [Source 2] then [Source 1] and again [Source 2]."
    citations = g._extract_citations(answer, retrieved_docs)

    assert [c["source_number"] for c in citations] == [1, 2]


def test_extract_citations_ignores_out_of_range_sources():
    g = AnswerGenerator()
    retrieved_docs = [
        {
            "content": "A",
            "metadata": {
                "source": "doc.pdf",
                "page": 1,
                "chunk_id": "c0",
                "chunk_index": 0,
            },
        }
    ]

    answer = "Bad cite [Source 2], good cite [Source 1]."
    citations = g._extract_citations(answer, retrieved_docs)

    assert len(citations) == 1
    assert citations[0]["source_number"] == 1
    assert citations[0]["source"] == "doc.pdf"
    assert citations[0]["page"] == 1
    assert citations[0]["chunk_id"] == "c0"
    assert citations[0]["chunk_index"] == 0