from triple_hybrid_rag.config import RAGConfig
from triple_hybrid_rag.ingestion.ocr import OCRProcessor


def test_ocr_processor_uses_config_defaults() -> None:
    config = RAGConfig(
        rag_ocr_api_base="http://example.com/v1",
        rag_ocr_model="test-ocr",
        rag_ocr_timeout=12.0,
        rag_gundam_tiling_enabled=True,
        rag_gundam_tile_size=512,
        rag_gundam_overlap=64,
        rag_gundam_min_image_size=1200,
        rag_gundam_max_tiles=8,
        rag_gundam_merge_strategy="concat",
        rag_gundam_fuzzy_threshold=0.9,
        rag_deepseek_ocr_enabled=False,
    )

    processor = OCRProcessor(settings=config)

    assert processor.endpoint == "http://example.com/v1"
    assert processor.model == "test-ocr"
    assert processor.timeout == 12.0
    assert processor.gundam_config.enabled is True
    assert processor.gundam_config.tile_size == 512
    assert processor.gundam_config.overlap == 64
    assert processor.gundam_config.min_image_size == 1200
    assert processor.gundam_config.max_tiles == 8
    assert processor.gundam_config.merge_strategy == "concat"
    assert processor.gundam_config.fuzzy_threshold == 0.9


def test_ocr_processor_deepseek_overrides() -> None:
    config = RAGConfig(
        rag_deepseek_ocr_enabled=True,
        rag_deepseek_ocr_api_base="http://deepseek.local/v1",
        rag_deepseek_ocr_model="deepseek-ocr-test",
    )

    processor = OCRProcessor(settings=config)

    assert processor.endpoint == "http://deepseek.local/v1"
    assert processor.model == "deepseek-ocr-test"
