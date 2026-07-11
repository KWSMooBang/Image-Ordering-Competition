from src.caption_augmented.config import DEFAULT_ORDER_MODEL
from src.caption_augmented.model import QWEN35_LOADER, QWEN3_VL_LOADER, apply_peft_adapter, qwen_model_loader_name


def test_default_order_model_is_qwen35_4b():
    assert DEFAULT_ORDER_MODEL == "Qwen/Qwen3.5-4B"


def test_qwen35_models_use_auto_multimodal_loader():
    assert qwen_model_loader_name("Qwen/Qwen3.5-4B") == QWEN35_LOADER
    assert qwen_model_loader_name("Qwen/Qwen3.5-9B") == QWEN35_LOADER


def test_qwen3_vl_models_keep_legacy_loader():
    assert qwen_model_loader_name("Qwen/Qwen3-VL-8B-Instruct") == QWEN3_VL_LOADER


def test_apply_peft_adapter_noops_when_adapter_path_is_none():
    model = object()

    assert apply_peft_adapter(model, None) is model
