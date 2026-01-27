# ETHOS-ARES Inference Files

## File Locations

In order to add these files to the ETHOS-ARES code base, the relative path of each file will be listed with its name.

---

## Basic File Descriptions

### `src/ethos/inference/run_inference_cached_split.py`
Manages the logic regarding dataset partitioning, resuming inference, assigning timelines to inference workers, and managing i/o logic.

### `src/ethos/model_kvcache.py`
Altered model class that allows for KV caching. Necessary for both `inference_cached.py` and `inference_outcome_exclusion.py`.

### `src/ethos/inference/inference_cached.py`
Handles inference to generate M0 and M1 outcome predictions.

### `src/ethos/inference/inference_outcome_exclusion.py`
Separate file that handles M2 outcome predictions and deals with altered next token logic.

### `src/ethos/configs/inference_cached.yaml`
Configuration file that determines some parameters for an inference run.
