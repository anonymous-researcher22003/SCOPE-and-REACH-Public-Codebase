# SCOPE and REACH Public Codebase

This public repository includes the necessary code to reproduce the results of the SCOPE and REACH paper. Each sub-directory contains a separate README file that will outline how to reproduce the results generated in this codebase. Additionally, the results files necessary to run the notebook files is being submitted to Physionet for publication.
![alt text](https://github.com/anonymous-researcher22003/SCOPE-and-REACH-Public-Codebase/blob/main/SCOPE%20and%20REACH%20Code/figures/flowchart.png)


SCOPE and REACH Code/                                                                                                                                                                                               
  ├── scripts/                              # Directory containing ETHOS-ARES shell scripts
  │   ├── README.md                     
  │   ├── run_full_split_inference.sh       # Batched inference shell script
  │   ├── run_tokenization.sh               # Tokenization shell script
  │   └── run_training.sh                   # Training shell script
  │
  ├── inference/                            # Directory containing all the modified inference methods
  │   ├── README.md                  
  │   ├── README.txt                        # ________________________________
  │   ├── inference_cached.py               # SCOPE and Monte Carlo Inference File
  │   ├── inference_cached.yaml             # Inference config file
  │   ├── inference_outcome_exclusion.py    # REACH inference file
  │   ├── model_kvcache.py                  # Modified model architecture to allow for KV-caching
  │   └── run_inference_cached_split.py     # Inference manager that calls the above inference files
  │
  ├── notebooks/                            # Notebooks used to generate the figures.
  │   ├── README.md                        
  │   ├── icu_admission.ipynb               # Notebook that generates figures regarding ICU Admission prediction task
  │   ├── metrics.py                        # Modified metrics python file that enables our methods and subsampling
  │   └── mortality.ipynb                   # Notebook that generates figures regarding ICU Admission prediction task
  │
  └── logs/                                 # Logs for transparent run-time improvements
      ├── README.md                        
      ├── ethos_infer.log                   # Partially run standard inference log
      └── ethos_M2_ed_4334712_*.log (x32)   # All patient mortality inference run using caching
