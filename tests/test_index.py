"""Tests for sigma-index inverted index."""

from sigma_index.index import InvertedIndex, tokenize


def test_tokenize_snake_case():
    assert tokenize("get_user_name") == ["get", "user", "name"]


def test_tokenize_camel_case():
    assert tokenize("getUserName") == ["get", "user", "name"]


def test_tokenize_filters_stop_words():
    tokens = tokenize("this is a test of the system")
    assert "this" not in tokens
    assert "test" in tokens
    assert "system" in tokens


def test_tokenize_single_char_filtered():
    tokens = tokenize("a b c def")
    assert tokens == ["def"]


def test_add_and_search():
    idx = InvertedIndex()
    idx.add_document("server.py", "def start_server():\n    listen(8080)")
    idx.add_document("client.py", "def connect_client():\n    send(data)")

    results = idx.search("server start")
    assert len(results) >= 1
    assert results[0].doc_id == "server.py"


def test_search_empty_index():
    idx = InvertedIndex()
    results = idx.search("anything")
    assert results == []


def test_search_no_match():
    idx = InvertedIndex()
    idx.add_document("a.py", "def foo(): pass")
    results = idx.search("zzzznonexistent")
    assert results == []


def test_remove_document():
    idx = InvertedIndex()
    idx.add_document("a.py", "def foo(): pass")
    assert idx.doc_count == 1
    assert idx.remove_document("a.py")
    assert idx.doc_count == 0
    assert not idx.remove_document("a.py")


def test_update_document():
    idx = InvertedIndex()
    idx.add_document("a.py", "def old_function(): pass")
    idx.add_document("a.py", "def new_function(): pass")
    assert idx.doc_count == 1

    results = idx.search("old_function")
    assert len(results) == 0

    results = idx.search("new_function")
    assert len(results) == 1


def test_tfidf_ranking():
    idx = InvertedIndex()
    # Doc with "server" appearing many times should rank higher
    idx.add_document("focused.py", "server server_config server_start server_stop")
    idx.add_document("mixed.py", "server client database config logger")

    results = idx.search("server")
    assert len(results) == 2
    assert results[0].doc_id == "focused.py"
    assert results[0].score > results[1].score


def test_doc_filter():
    idx = InvertedIndex()
    idx.add_document("a.py", "def foo(): pass")
    idx.add_document("b.py", "def foo(): pass")

    results = idx.search("foo", doc_filter={"a.py"})
    assert len(results) == 1
    assert results[0].doc_id == "a.py"


def test_get_document():
    idx = InvertedIndex()
    idx.add_document("a.py", "content here")
    assert idx.get_document("a.py") == "content here"
    assert idx.get_document("missing.py") is None


def test_get_stats():
    idx = InvertedIndex()
    idx.add_document("a.py", "def foo(): bar baz")
    stats = idx.get_stats()
    assert stats["documents"] == 1
    assert stats["unique_tokens"] > 0
    assert stats["total_tokens"] > 0


def test_snippet_extraction():
    idx = InvertedIndex()
    idx.add_document("long.py", "line one\ndef target_function():\n    return 42\nline four")
    results = idx.search("target_function")
    assert len(results) == 1
    assert "target_function" in results[0].snippet


def test_multi_token_query():
    idx = InvertedIndex()
    idx.add_document("match.py", "def http_response_handler(): process()")
    idx.add_document("partial.py", "def http_client(): connect()")
    idx.add_document("none.py", "def database_query(): select()")

    results = idx.search("http response handler")
    assert results[0].doc_id == "match.py"
