# Topic Modeling with Wasserstein Autoencoders

Source code for `Nan, F., Ding, R., Nallapati, R., & Xiang, B. (2019, July). Topic Modeling with Wasserstein Autoencoders. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (pp. 6345-6381).`

## Setup:
* Download or clone the w-lda repo. Denote the repo location as `SOURCE_DIR`.
* Create a conda environment and install necessary packages:
  * `conda create --name w-lda python=3.6` and `conda activate w-lda`
  * install `mxnet-cu100` (or `mxnet-cu90` depending on CUDA version), `matplotlib`, `scipy`, `scikit_learn`, `tqdm`, `nltk`

## Pre-process data:
We provide a script to process the `Wikitext-103` dataset, which can be downloaded [here](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/). 

* `export PYTHONPATH="$PYTHONPATH:<SOURCE_DIR>"`
* from `SOURCE_DIR`, run `python examples/domains/wikitext103_wae.py`

This will download the dataset and store the pre-processed data under `SOURCE_DIR/data/wikitext-103` (note the pre-processing may take a while). 

## Training the model:
* from `SOURCE_DIR`, run `./examples/gpu0.sh`

The result is saved under `SOURCE_DIR/examples/results`. In particular, the top words of the topics are saved under `eval_record.p` under the keys `Top Words` and `Top Words2`.
`Top Words2` are top words based on ranking the decoder matrix weights; `Top Words` are the top words based on the decoder output for each topic (the corresponding column of the decoder matrix plus the offset). 
Note that in order to evaluate NPMI scores, a separate server process needs to run `npmi_calc.py`, which would require the
dictionary and inverted index files for the Wikipedia corpus. We do not currently provide these files so the NPMIs are set to 0's. 
However, readers can refer to other open source packages such as [this](https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Evaluate%20Topic%20Models.ipynb) for evaluation.

## License

This project is licensed under the Apache-2.0 License.

