BASIC FILE DESCRIPTIONS

run_inference_cahced_split.py

Manages the logic regarding dataset partitioning, resuming inference, assigning timelines to inference workers, and managing i/o logic

model_kvcache.py

Altered model class that allows for KV caching. Necessary for both inference_cached.py and inference_outcome_exclusion.py.

inference_cached.py

Handles inference to generate M0 and M1 outcome predictions. 

inference_outcome_exclusion.py

Separate file that handles M2 outcome predictions and deals with altered next token logic

inference_cached.yaml

Configuration file that determines some parameters for an inference run.