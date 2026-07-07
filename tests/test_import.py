import zoom_search


def test_import_zoom_search() -> None:
    assert zoom_search.__version__ == "0.1.3"
    assert hasattr(zoom_search, "SearchRequest")
