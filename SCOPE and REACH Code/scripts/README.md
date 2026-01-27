# ETHOS-ARES Shell Scripts

## File Locations

In order to add these files to the ETHOS-ARES code base, they should be added to the root of the ETHOS-ARES directory. Additionally, the following additions to pyproject.toml should be made:

`...`
`[project scripts]`
`...`
`ethos_infer_split = "ethos.inference.run_inference_cached_split:main"`

`[tool.setuptools.package-data]`
`...`

---

## Basic File Descriptions
These are the exact files used to generate our results. Since they are tailored to our SLURM cluster, adjustments will likely need to be made to run these on another platform.


### `run_fulltokenization.sh`
Example shell script used to run ETHOS-ARES tokenization (needs to be altered to run on another cluster)

### `run_training.sh`
Example shell script used to run ETHOS-ARES training (needs to be altered to run on another cluster)

### `run_full_split_inference.sh`
Example shell script used to run inference on all tasks. (NOTE: In order to decide which estimator, M1 or M2, is used, you must edit the inference-cached.yaml file)