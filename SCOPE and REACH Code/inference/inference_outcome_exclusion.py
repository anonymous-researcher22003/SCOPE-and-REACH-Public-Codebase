"""
Separate inference file used for M2 predictions.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from copy import copy
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple

import torch as th

from ..constants import SpecialToken as ST
from ..model_kvcache import GPT2LMNoBiasModelKV, convert_pretrained_to_kv_model
from ..utils import load_model_checkpoint, setup_torch
from ..vocabulary import Vocabulary
from .constants import Reason, Task
from .utils import create_loader, get_dataset_cls, get_token_time


def spawn_inference_worker_outcome_exclusion(
    job_queue: Queue,
    model_fp: str | Path,
    task: Task,
    dataset_kwargs: dict,
    progress_queue: Queue,
    temperature: float = 1.0,
    rep_num: int = 1,
    device: str = "cuda",
    no_compile: bool = False,
    save_generated_tokens: bool = False,
    save_logits: bool = True,
    skip_size = 64
):
    if "cuda" in device:
        th.cuda.set_device(device)
        th.set_float32_matmul_precision("high")
    autocast_context = setup_torch(device, dtype="bfloat16" if "cuda" in device else "float32")

    # Load model and convert to KV-cache version
    model, _ = load_model_checkpoint(model_fp, map_location=device)
    model_kv = convert_pretrained_to_kv_model(model)
    model_kv.to(device)

    # IMPORTANT: Original ETHOS-ARES database does not activate eval mode for inference
    # Leaving eval mode off during inference had a positive effect on downstream prediction accuracy
    # in our testing

    #model_kv.eval()

    model_kv = th.compile(model_kv, disable = no_compile)
    dataset_cls = get_dataset_cls(task)
    dataset = dataset_cls(**dataset_kwargs)

    max_timeline_size = dataset_kwargs["n_positions"]
    ctx_size = dataset.context_size
    vocab: Vocabulary = dataset.vocab

    data_loader = create_loader(job_queue, dataset)

    stop_stokens = dataset.stop_stokens
    stop_tokens = th.tensor(vocab.encode(stop_stokens), dtype=th.long, device=device)
    time_limit = th.tensor(dataset.time_limit.total_seconds() * 1e6)

    for sample_idx, (timeline, ground_truth) in enumerate(data_loader):
        ctx = None
        if isinstance(timeline, tuple):
            ctx, timeline = tuple(t.to(device, non_blocking=True) for t in timeline)
            ctx = ctx.repeat(rep_num, 1)
        else:
            timeline = timeline.to(device, non_blocking=True)
            
        timeline = timeline.repeat(rep_num, 1)
        process_batch_outcome_exclusion(
            model_kv,
            timeline,
            ctx,
            ground_truth,
            progress_queue,
            task,
            vocab,
            stop_tokens,
            stop_stokens,
            time_limit,
            temperature,
            device,
            save_generated_tokens,
            save_logits,
            autocast_context,
            max_timeline_size,
            ctx_size,
            skip_size
        )

        # Free memory after each sample
        del timeline, ctx


@th.no_grad()
def process_batch_outcome_exclusion(
    model: GPT2LMNoBiasModelKV,
    timeline: th.Tensor,
    ctx: Optional[th.Tensor],
    ground_truth: dict,
    progress_queue: Queue,
    task: Task,
    vocab: Vocabulary,
    stop_tokens: th.Tensor,
    stop_stokens: list,
    time_limit: th.Tensor,
    temperature: float,
    device: str,
    save_generated_tokens: bool,
    save_logits: bool,
    autocast_context,
    max_timeline_size: int,
    ctx_size: int,
    skip_size: int
):
    batch_size = timeline.size(0)
    gen_token_num = 0
    offset = 0
    gen_times = th.zeros(batch_size, dtype=th.float64, device=device)
    generated_tokens = [] if save_generated_tokens else None

    probs_list = th.zeros(batch_size, len(stop_tokens), device=device)

    all_logits_list = [[] for _ in range(batch_size)] if save_logits else None

    # Initialize cumulative outcome probability tracking
    # Track ∏(1 - p_i) for each step, final result is 1 - ∏(1 - p_i)
    cumulative_non_outcome_prob = th.ones(batch_size, dtype=th.float64, device=device)

    # Get outcome token to exclude from task definition
    outcome_token = ground_truth.get("outcome_token")
    if outcome_token is None:
        raise ValueError(f"outcome_token not found in ground_truth for task {task}. "
                        f"This task may not support outcome exclusion inference.")
    outcome_token_id = vocab.encode([outcome_token])[0]

    # KV cache tracking
    past_key_values = None
    
    while timeline.size(0) > 0:
        with autocast_context:
            if past_key_values is None:
                output = model(timeline, use_cache=True)
                past_key_values = output.past_key_values
                logits = output.logits[:, -1, :] / temperature
            else:
                output = model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = output.past_key_values
                logits = output.logits[:, -1, :] / temperature 
            # OUTCOME EXCLUSION LOGIC:
            # 1. Compute FULL probability distribution (before masking)
            probs_full = th.softmax(logits, dim=-1)

            # 2. Extract outcome token probability
            p_outcome = probs_full[:, outcome_token_id]

            # 3. Update cumulative non-outcome probability: multiply by (1 - p_i)
            cumulative_non_outcome_prob *= (1 - p_outcome)

            # 4. Mask outcome token by setting logit to -inf
            logits[:, outcome_token_id] = -float('inf')
            
            # 5. Re-compute probabilities with masked outcome token
            probs = th.softmax(logits, dim=-1)

            # 6. Sample from masked distribution (outcome token cannot be chosen)
            if temperature == 0:
                next_token = th.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = th.multinomial(probs, num_samples=1)

            # Store logits if requested (save masked logits)
            if save_logits:
                for i in range(timeline.size(0)):
                    all_logits_list[i].append(logits[i])
        if generated_tokens is not None:
            generated_tokens.append(next_token)

        # If sequence is longer than max window length, skip skip_size tokens and reset cache
        if not offset and timeline.size(1) >= max_timeline_size:
            offset = skip_size
            past_key_values = None

        if ctx is not None:
            new_timeline = (timeline[:, offset:], next_token)
        else:
            new_timeline = (
                timeline[:, :ctx_size],
                timeline[:, ctx_size + offset:],
                next_token,
            )
        timeline = th.cat(new_timeline, dim=1)
        offset = 0
        gen_token_num += 1

        new_token = next_token.cpu().view(-1)
        gen_times += get_token_time(new_token, vocab).to(device)

        # Check completion conditions 
        completed_this_iter = th.isin(new_token, stop_tokens.cpu()) | (gen_times.cpu() > time_limit.item())

        if task == Task.DRG_PREDICTION or (task == Task.SOFA_PREDICTION and gen_token_num == 3):
            completed_this_iter[:] = True

        # Update probability lists (Note that these are not valid M1 predictions. Mostly used for debugging)
        probs_list = probs_list + probs[:, stop_tokens]

        if not completed_this_iter.any():
            continue

        # Process completed sequences
        for idx, i in enumerate(th.nonzero(completed_this_iter, as_tuple=False)):
            i = i.item()
            stop_reason = Reason.GOT_TOKEN
            actual_token = next_token[i].item()

            token_time = gen_times[i]
            if token_time > time_limit.item():
                stop_reason = Reason.TIME_LIMIT

            if th.isinf(token_time):
                actual_stoken = str(actual_token)
                print("KEY ERROR STOP STOP STOP STOP")
                print(str(actual_token))
                stop_reason = Reason.KEY_ERROR
                token_time = None
            else:
                actual_stoken = vocab.decode(actual_token)
                token_time = round(token_time.item())

            gt = copy(ground_truth)
            # Calculate cumulative outcome probability: 1 - ∏(1 - p_i)
            cumulative_outcome_prob = (1 - cumulative_non_outcome_prob[i]).item()

            results = {
                "expected": gt.pop("expected"),
                "actual": actual_stoken,
                "stop_reason": stop_reason,
                "actual_prob": probs[i, actual_token].item(),
                **dict(zip(stop_stokens, probs_list[i].tolist())),
                "true_token_time": gt.pop("true_token_time"),
                "token_time": token_time,
                "true_token_dist": gt.pop("true_token_dist"),
                "token_dist": gen_token_num,
                "cumulative_outcome_prob": cumulative_outcome_prob,  # UNIQUE: cumulative probability of excluded outcome
                "outcome_token": gt.pop("outcome_token"),  # UNIQUE: which token was excluded
                **gt,
            }

            if generated_tokens is not None:
                results["generated_tokens"] = [tokens[i].item() for tokens in generated_tokens]

            if save_logits and all_logits_list is not None:
                results["generated_logits"] = th.stack(all_logits_list[i]).cpu().float().numpy()

            progress_queue.put(results)
        if completed_this_iter.all():
            break
        # Remove completed sequences from the batch
        not_completed_mask = ~completed_this_iter
        not_completed_mask_device = not_completed_mask.to(device)

        timeline = timeline[not_completed_mask_device, :]
        next_token = next_token[not_completed_mask_device, :]
        probs_list = probs_list[not_completed_mask_device, :]
        gen_times = gen_times[not_completed_mask_device]
        cumulative_non_outcome_prob = cumulative_non_outcome_prob[not_completed_mask_device]
        if ctx is not None:
            ctx = ctx[not_completed_mask_device, :]
        if generated_tokens is not None:
            generated_tokens = [tokens[not_completed_mask_device, :] for tokens in generated_tokens]
        if save_logits and all_logits_list is not None:
            all_logits_list = [all_logits_list[i] for i in range(len(all_logits_list)) if not_completed_mask[i]]

        # Update KV cache to remove completed sequences
        if past_key_values is not None:
            past_key_values = tuple(
                (k[not_completed_mask_device, :, :, :], v[not_completed_mask_device, :, :, :])
                for k, v in past_key_values
            )