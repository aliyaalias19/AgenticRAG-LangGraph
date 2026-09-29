"""Merge the LoRA adapter into the base model and quantise to AWQ.

Order is not negotiable: an adapter cannot be AWQ-quantised while it is still
an adapter, because AWQ calibrates activation scales over the weights it is
going to serve. The sequence is

    4-bit QLoRA training  ->  merge into fp16  ->  AWQ quantise  ->  serve

Merging a 4-bit-trained adapter into fp16 weights shifts quality slightly,
and AWQ shifts it again. Both shifts land in the artefact that gets served,
which is why held-out accuracy must be measured on the final AWQ model and
never on the training checkpoint. Reporting the adapter's score for a model
you serve quantised is the most common way this benchmark gets quietly
overstated.
"""

import argparse
import json
import sys
from pathlib import Path

from agentic_rag.obs.logging import configure_logging, get_logger

logger = get_logger(__name__)


def merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path) -> None:
    """Merge LoRA weights into the base model and save in fp16."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("merge_started", base=base_model, adapter=str(adapter_dir))

    # Loaded in fp16, not 4-bit: merging into quantised weights would fold the
    # adapter into an already-lossy tensor and compound the error.
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(output_dir))
    logger.info("merge_completed", output=str(output_dir))


def quantize_awq(
    merged_dir: Path,
    output_dir: Path,
    calibration_file: Path | None = None,
    group_size: int = 128,
) -> None:
    """Quantise the merged model to 4-bit AWQ.

    AWQ protects the roughly one percent of weight channels that carry the
    largest activations, scaling them before rounding so their precision
    survives. Calibration data decides which channels those are, so it should
    resemble production traffic; calibrating on generic web text and serving
    Kubernetes documentation leaves accuracy on the table.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": 4,
        "version": "GEMM",
    }

    logger.info("awq_started", model=str(merged_dir), **config)
    model = AutoAWQForCausalLM.from_pretrained(str(merged_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))

    calibration = None
    if calibration_file and calibration_file.is_file():
        calibration = [
            json.loads(line)["text"]
            for line in calibration_file.read_text("utf-8").splitlines()
            if line.strip()
        ]
        logger.info("awq_calibration_loaded", samples=len(calibration))

    model.quantize(tokenizer, quant_config=config, calib_data=calibration)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("awq_completed", output=str(output_dir))


def build_calibration_set(dataset_path: Path, destination: Path, limit: int) -> None:
    """Write in-domain calibration text drawn from the training set."""
    from agentic_rag.train.dataset import read_jsonl

    examples = read_jsonl(dataset_path)[:limit]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            text = f"{example.instruction}\n\n{example.response}"
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    logger.info("calibration_written", path=str(destination), samples=len(examples))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged", type=Path, default=Path("data/models/merged"))
    parser.add_argument("--awq", type=Path, default=Path("data/models/awq"))
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()
    if not args.skip_merge:
        merge_adapter(args.base_model, args.adapter, args.merged)
    if not args.skip_quantize:
        quantize_awq(args.merged, args.awq, args.calibration, args.group_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
