# SCOPE and REACH Public Codebase
The Markov chain experiments interactive notebook is located [here](https://colab.research.google.com/drive/1PURIj0iZGcZITVx9FsAvqixKn6iF1_CE#scrollTo=yilfxvKE87bK) 
```
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
```

![alt text](https://anonymous.4open.science/r/SCOPE-and-REACH-Public-Codebase-1185/SCOPE%20and%20REACH%20Code/figures/flowchart.png)
* Welcome to the public repository for the SCOPE and REACH paper! 
* This repository includes the necessary code to reproduce the results of the SCOPE and REACH paper. 
* Each sub-directory contains a separate README file that will outline how to reproduce the results generated in this codebase.
* The results files necessary to run the notebook files are being submitted to Physionet for publication.


