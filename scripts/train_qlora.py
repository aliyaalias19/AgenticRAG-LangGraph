"""QLoRA supervised fine-tuning of Llama 3.1 8B.

Quantised LoRA keeps an 8B model trainable on a single 24GB card. The base
weights are frozen in 4-bit NF4; only the low-rank adapters carry gradients,
so optimiser state is measured in megabytes rather than tens of gigabytes.

NF4 rather than plain int4 because the format is information-theoretically
matched to the roughly normal distribution of trained weights: each of the 16
levels covers an equal share of the probability mass instead of an equal
slice of the range, which is where most of the quality would otherwise go.

Double quantisation compresses the per-block quantisation constants too,
saving about 0.4 bits per parameter -- roughly 400MB at this scale, which is
often the difference between fitting and not.

    python scripts/train_qlora.py --config configs/qlora_r16.json

Every run logs its full hyperparameter set and its evaluation loss to MLflow,
so a sweep is reconstructable rather than remembered.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agentic_rag.config.settings import get_settings
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.train.dataset import TRAIN_FILENAME, read_jsonl

logger = get_logger(__name__)


@dataclass
class LoRAConfig:
    """One point in the LoRA hyperparameter space."""

    name: str = "r16_a32"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @property
    def scaling(self) -> float:
        """Return alpha/rank, the factor applied to the adapter output.

        Holding this constant while sweeping rank is what separates "more
        capacity" from "louder adapter"; changing both at once confounds the
        two and makes the sweep uninterpretable.
        """
        return self.alpha / self.rank


@dataclass
class TrainingConfig:
    """Full training configuration for one run."""

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "data/models/qlora"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    learning_rate: float = 2e-4
    epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    seed: int = 42

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    gradient_checkpointing: bool = True
    eval_fraction: float = 0.1

    @property
    def effective_batch_size(self) -> int:
        """Return the true optimiser batch size.

        Gradient accumulation is what lets a 24GB card train at an effective
        batch of 16 while only ever holding 2 sequences in memory.
        """
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lora"]["target_modules"] = list(self.lora.target_modules)
        payload["lora"]["scaling"] = self.lora.scaling
        payload["effective_batch_size"] = self.effective_batch_size
        return payload

    @staticmethod
    def from_file(path: Path) -> "TrainingConfig":
        """Load a configuration from JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        lora_payload = payload.pop("lora", {})
        if "target_modules" in lora_payload:
            lora_payload["target_modules"] = tuple(lora_payload["target_modules"])
        lora_payload.pop("scaling", None)
        payload.pop("effective_batch_size", None)
        return TrainingConfig(lora=LoRAConfig(**lora_payload), **payload)


def build_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    """Load the quantised base model and attach LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantisation = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
    )

    logger.info("loading_base_model", model=config.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantisation,
        device_map="auto",
        torch_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )

    adapter = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, adapter)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        # Llama ships no pad token. Reusing EOS is standard; the collator
        # masks pad positions out of the loss, so it costs nothing.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def train(config: TrainingConfig, dataset_path: Path) -> dict[str, Any]:
    """Run one fine-tuning job and return its metrics."""
    import mlflow
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    examples = read_jsonl(dataset_path)
    logger.info("dataset_loaded", examples=len(examples), path=str(dataset_path))

    records = [{"messages": e.to_messages()} for e in examples]
    dataset = Dataset.from_list(records).train_test_split(
        test_size=config.eval_fraction, seed=config.seed
    )

    model, tokenizer = build_model_and_tokenizer(config)
    output_dir = Path(config.output_dir) / config.lora.name

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        seed=config.seed,
        report_to=[],
    )

    with mlflow.start_run(run_name=f"qlora-{config.lora.name}"):
        mlflow.log_params(
            {
                "base_model": config.base_model,
                "lora_rank": config.lora.rank,
                "lora_alpha": config.lora.alpha,
                "lora_scaling": config.lora.scaling,
                "lora_dropout": config.lora.dropout,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "effective_batch_size": config.effective_batch_size,
                "max_sequence_length": config.max_sequence_length,
                "quant_type": config.bnb_4bit_quant_type,
                "double_quant": config.bnb_4bit_use_double_quant,
                "train_examples": len(dataset["train"]),
            }
        )

        trainer = SFTTrainer(
            model=model,
            args=arguments,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
        )
        result = trainer.train()
        metrics = trainer.evaluate()

        mlflow.log_metrics(
            {
                "train_loss": result.training_loss,
                "eval_loss": metrics.get("eval_loss", 0.0),
                "train_runtime_seconds": result.metrics.get("train_runtime", 0.0),
            }
        )

        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        (output_dir / "training_config.json").write_text(
            json.dumps(config.as_dict(), indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(output_dir / "training_config.json"))

    logger.info("training_completed", output_dir=str(output_dir), **metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved configuration and exit"
    )
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()

    config = TrainingConfig.from_file(args.config) if args.config else TrainingConfig()
    if args.rank is not None:
        config.lora.rank = args.rank
    if args.alpha is not None:
        config.lora.alpha = args.alpha
    if args.dropout is not None:
        config.lora.dropout = args.dropout
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epochs is not None:
        config.epochs = args.epochs
    config.lora.name = f"r{config.lora.rank}_a{config.lora.alpha}"

    if args.dry_run:
        print(json.dumps(config.as_dict(), indent=2))
        return 0

    dataset_path = args.dataset or settings.paths.evalsets_dir / TRAIN_FILENAME
    train(config, dataset_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
