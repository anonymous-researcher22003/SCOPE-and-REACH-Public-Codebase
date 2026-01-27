"""
Replacement for run_inference.py with KV-cache support and dataset partitioning.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from multiprocessing import Manager, Process, set_start_method
from pathlib import Path
from queue import Empty

import hydra
import numpy as np
import torch as th
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from .constants import Task
from .inference_cached import spawn_inference_worker_cached
from .inference_outcome_exclusion import spawn_inference_worker_outcome_exclusion
from .utils import (
    evaluate_dataset_subset,
    filter_completed_indices,
    format_big_number,
    get_completed_timeline_ids,
    producer,
    write_results_to_parquet_chunk,
)
from .inference import get_dataset_cls


def get_partition_indices(total_size: int, num_parts: int, part_idx: int) -> np.ndarray:
    # Gets the indices for a specific partition of the dataset
    if part_idx >= num_parts:
        raise ValueError(f"part_idx ({part_idx}) must be less than num_parts ({num_parts})")
    
    # Use array_split for even distribution
    all_indices = np.arange(total_size)
    partitions = np.array_split(all_indices, num_parts)
    return partitions[part_idx]


@hydra.main(version_base=None, config_path="../configs", config_name="inference_cached")
def main(cfg: DictConfig):
    task = Task(cfg.task)
    input_dir = Path(cfg.input_dir)

    model_checkpoint = th.load(cfg.model_fp, map_location="cpu", mmap=True, weights_only=False)

    model_config = model_checkpoint["model_config"]
    n_positions = (
        model_config.decoder.n_positions
        if model_config.is_encoder_decoder
        else model_config.n_positions
    )
    
    # Extract partitioning parameters before passing to dataset
    from omegaconf import OmegaConf
    if cfg.dataset_kwargs:
        dataset_kwargs_dict = OmegaConf.to_container(cfg.dataset_kwargs, resolve=True)
        if not isinstance(dataset_kwargs_dict, dict):
            dataset_kwargs_dict = {}
    else:
        dataset_kwargs_dict = {}
    
    num_parts = dataset_kwargs_dict.pop("num_parts", None)
    part_idx = dataset_kwargs_dict.pop("part_idx", None)
    
    logger.info(f"Partitioning config: num_parts={num_parts}, part_idx={part_idx}")
    logger.info(f"Dataset kwargs after pop: {dataset_kwargs_dict}")
    
    dataset_kwargs = {
        "input_dir": input_dir,
        "n_positions": n_positions,
        "is_encoder_decoder": model_config.is_encoder_decoder,
    }
    dataset_kwargs.update(dataset_kwargs_dict)
    
    logger.info(f"Final dataset kwargs keys: {list(dataset_kwargs.keys())}")

    dataset_cls = get_dataset_cls(task)
    start_time = time.time()
    dataset = dataset_cls(**dataset_kwargs)
    logger.info(f"{dataset} initialized in {time.time() - start_time:.0f}s.")

    if len(stop_stokens := dataset.stop_stokens) > 10:
        stop_stokens = stop_stokens[:10] + ["..."]
    logger.info(f"Stop tokens: {', '.join(stop_stokens)}")
    logger.info(f"Time limit: {dataset.time_limit}")

    # Handle dataset partitioning for array jobs
    if num_parts is not None and part_idx is not None:
        logger.info(f"Dataset partitioning enabled: part {part_idx + 1} of {num_parts}")
        partition_indices = get_partition_indices(len(dataset), num_parts, part_idx)
        logger.info(f"  - This partition contains {len(partition_indices)} samples")
        logger.info(f"  - Index range: {partition_indices[0]} to {partition_indices[-1]}")
    else:
        partition_indices = None

    # Evaluate subset (applies to partition if partitioning is enabled)
    if partition_indices is not None:
        # When partitioning, n_samples applies to the partition
        effective_dataset_size = len(partition_indices)
    else:
        effective_dataset_size = len(dataset)
    
    n_samples, subset_suffix = evaluate_dataset_subset(dataset, cfg.subset)
    
    if partition_indices is not None:
        # Adjust n_samples to not exceed partition size
        n_samples = min(n_samples, len(partition_indices))
    
    logger.info(
        f"Number of samples: {n_samples:,} ({n_samples / len(dataset):.2%} of full dataset)."
        + (f" Full dataset size: {len(dataset):,}." if subset_suffix or partition_indices is not None else "")
        + f" Number of repetitions: {cfg.rep_num}"
    )
    
    result_dir = Path(cfg.output_dir + subset_suffix)

    if cfg.temperature != 1.0:
        result_dir = result_dir.with_name(f"{result_dir.name}_temp{cfg.temperature}")

    if cfg.output_fn is not None:
        result_dir /= cfg.output_fn
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to '{result_dir}'")

    # Get cached inference parameters
    save_logits = cfg.get("save_logits", True)

    logger.info(f"Cached inference enabled:")
    logger.info(f"  - Save logits: {save_logits}")
    logger.info(f"  - Generation continues until stop conditions (stop tokens or time limit)")

    # Resume functionality: check for completed timelines
    id_field = "hadm_id" if hasattr(dataset, "is_mimic") and dataset.is_mimic else "patient_id"
    completed_ids, num_completed = get_completed_timeline_ids(result_dir, id_field=id_field, expected_rep_num=cfg.rep_num)
    num_completed = num_completed * cfg.result_chunk_size
    
    # Determine the pool of indices to sample from
    if partition_indices is not None:
        available_indices = partition_indices
    else:
        available_indices = np.arange(len(dataset))
    # Filter available indices to get only uncompleted ones
    if completed_ids:
        logger.info(f"Resume mode: Found {len(completed_ids)} completed timelines (identified by '{id_field}')")
        remaining_indices = filter_completed_indices(dataset, available_indices, completed_ids, id_field=id_field)
        logger.info(f"  - Available timelines: {len(available_indices)}")
        logger.info(f"  - Completed: {len(completed_ids)}")
        logger.info(f"  - Remaining: {len(remaining_indices)}")

        # Sample from remaining indices
        if len(remaining_indices) == 0:
            logger.info("All timelines already completed. Nothing to do.")
            return

        np.random.seed(cfg.seed)
        # Adjust n_samples once more if necessary
        actual_n_samples = min(n_samples, len(remaining_indices))
        if actual_n_samples < n_samples:
            logger.warning(f"Only {actual_n_samples} timelines remaining (requested {n_samples})")
        indices = np.random.choice(remaining_indices, actual_n_samples, replace=False)
    else:
        logger.info(f"No existing results found in {result_dir}, starting fresh")
        np.random.seed(cfg.seed)
        actual_n_samples = min(n_samples, len(available_indices))
        indices = np.random.choice(available_indices, actual_n_samples, replace=False)

    # Use actual number of indices for chunking
    chunk_num = len(indices) // cfg.chunksize if len(indices) > 0 else 1
    subsets = [subset_indices for subset_indices in np.array_split(indices, chunk_num)]

    if cfg.device == "cuda":
        num_proc = cfg.n_jobs * cfg.n_gpus
    elif cfg.device == "cpu":
        num_proc = cfg.n_jobs
    else:
        raise ValueError(f"Unknown device: {cfg.device}, must be 'cpu' or 'cuda'")
    if num_proc > len(subsets):
        logger.warning(
            f"Number of processes ({num_proc}) is larger than the number of subsets "
            f"({len(subsets)}). Launching only {len(subsets)} processes."
        )
        num_proc = len(subsets)

    set_start_method("spawn")
    with Manager() as mgr:
        job_queue = mgr.Queue(maxsize=num_proc * 2)
        progress_queue = mgr.Queue()

        processes = [Process(target=producer, args=(subsets, job_queue, num_proc), name="producer")]
        processes.extend(
            Process(
                target=spawn_inference_worker_cached if (not cfg.M2_estimator) else spawn_inference_worker_outcome_exclusion,
                args=(
                    job_queue,
                    cfg.model_fp,
                    task,
                    dataset_kwargs,
                    progress_queue,
                    cfg.temperature,
                    cfg.rep_num,
                    "cpu" if cfg.device == "cpu" else f"cuda:{i % cfg.n_gpus}",
                    cfg.no_compile,
                    cfg.save_generated_tokens,
                    save_logits,
                ),
                name=f"Process_{i}",
            )
            for i in range(num_proc)
        )

        for p in processes:
            p.start()

        results, generated_tokens = [], 0
        total_samples = len(indices) * cfg.rep_num
        
        # Include partition info in progress bar description
        if num_parts is not None and part_idx is not None:
            desc = f"Part {part_idx}/{num_parts}"
        else:
            desc = "Progress"
        
        progress_bar = tqdm(total=total_samples, desc=desc, unit="samples", smoothing=0.1)
        try:
            for _ in range(total_samples):
                result = progress_queue.get(timeout=cfg.timeout)

                # Handle logits separately if they exist (NOTE: not yet tested)
                if "generated_logits" in result and result["generated_logits"] is not None:
                    pass

                results.append(result)
                generated_tokens += result["token_dist"]
                progress_bar.set_postfix_str(
                    "total generated tokens: {}, {} tokens/s".format(
                        format_big_number(generated_tokens),
                        format_big_number(generated_tokens / progress_bar.format_dict["elapsed"]),
                    )
                )
                progress_bar.update()

                if len(results) >= cfg.result_chunk_size:
                    # Include partition index in filename to avoid collisions
                    chunk_id = progress_bar.format_dict["n"] + num_completed
                    if num_parts is not None and part_idx is not None:
                        chunk_id_str = f"part{part_idx:02d}_{chunk_id}"
                    else:
                        chunk_id_str = str(chunk_id)
                    print(f'WRITING RESULTS: {chunk_id_str}')
                    write_results_to_parquet_chunk(result_dir, results, chunk_id_str)
                    results = []

        except Empty:
            logger.error("Progress queue timed out.")
            for p in processes:
                if p.is_alive():
                    p.terminate()

        for p in processes:
            p.join()

    if results:
        chunk_id = progress_bar.format_dict["n"]
        if num_parts is not None and part_idx is not None:
            chunk_id_str = f"part{part_idx:02d}_{chunk_id}"
        else:
            chunk_id_str = str(chunk_id)
        write_results_to_parquet_chunk(result_dir, results, chunk_id_str)

    logger.info("Workers finished.")


if __name__ == "__main__":
    main()