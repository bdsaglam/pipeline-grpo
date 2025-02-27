import gc
import logging
import os
import re
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

log = logging.getLogger(__name__)

load_dotenv()

# System prompt and format definitions
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


# Dataset preparation
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    log.debug(
        "-" * 80 + "\n"
        f"Question:\n{q}\n"
        f"Answer:\n{answer[0]}\n"
        f"Response:\n{responses[0]}\n"
        f"Extracted:\n{extracted_responses[0]}\n"
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def clear_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()


def main(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_split: str = "train",
    batch_size: int = 8,
    num_generations: int = 4,
    use_vllm: bool = False,
    learning_rate: float = 5e-6,
    gradient_accumulation_steps: int = 4,
    flash_attn: bool = False,
    num_epochs: int = 1,
    save_steps: int = 100,
    out: Path = typer.Option("./tmp/outputs/"),
    hub_dir: Path = typer.Option("~/.cache/huggingface/tgi/local"),
    publish: bool = False,
    suffix: str = "-GRPO",
    run_name: Optional[str] = None,
    log_level: str = "INFO",
):
    """
    Train a model using GRPO on the GSM8K dataset.

    Args:
        model_name: Name of the model to use
        dataset_split: Dataset split to use for training
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_generations: Number of generations for GRPO
        num_epochs: Number of training epochs
        save_steps: Number of steps between model checkpoints
        use_flash_attention: Whether to use flash attention
        output_dir: Directory to save model checkpoints
        run_name: Name of the run for wandb
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    hub_dir = Path(hub_dir).resolve()
    hub_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging with the provided log level
    logging.basicConfig(level=log_level)

    # Setup environment
    clear_memory()

    # Load dataset
    dataset = get_gsm8k_questions(split=dataset_split)

    # Load model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": torch.cuda.current_device()},
    }

    if flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model_kwargs["use_cache"] = False

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # Configure training arguments
    output_dir = out / f"{model_name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        use_vllm=use_vllm,
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=num_epochs,
        save_steps=save_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Start training
    trainer.train()

    final_model_id = model_name.split("/", 1)[1] + suffix

    # Save locally
    log.info(f"Saving adapters locally: {final_model_id}")
    trainer.model.save_pretrained(hub_dir / final_model_id)
    tokenizer.save_pretrained(hub_dir / final_model_id)

    if publish:
        try:
            log.info(f"Pushing adapters to hub: {final_model_id}")
            trainer.model.push_to_hub(final_model_id)
            tokenizer.push_to_hub(final_model_id)
        except Exception as e:
            log.warning(f"Failed to push adapters to hub: {e}")

    # Merge adapters and save
    merged_model_id = final_model_id + "-merged"
    log.info(f"Merging adapters and saving locally: {merged_model_id}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(hub_dir / merged_model_id)
    tokenizer.save_pretrained(hub_dir / merged_model_id)

    if publish:
        try:
            log.info(f"Pushing merged model to hub: {merged_model_id}")
            merged_model.push_to_hub(hub_dir / merged_model_id)
            tokenizer.push_to_hub(hub_dir / merged_model_id)
        except Exception as e:
            log.warning(f"Failed to push merged model to hub: {e}")


if __name__ == "__main__":
    typer.run(main)
