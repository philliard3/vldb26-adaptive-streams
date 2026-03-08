A submission for VLDB 2026.

# Datasets

## ECG

We use a randomized order of items from the "dev" split of the ECG dataset from an open dataset, as determined by the RITA project.

https://github.com/ljmzlh/RITA

https://storage.googleapis.com/rita_resources/rita_dataset.tar.gz


Feifei Liu, Chengyu Liu, Lina Zhao, Xiangyu Zhang, Xiaoling Wu, Xiaoyan
Xu, Yulin Liu, Caiyun Ma, Shoushui Wei, Zhiqiang He, et al. 2018. An open
access database for evaluating the algorithms of electrocardiogram rhythm and
morphology abnormality detection. Journal of Medical Imaging and Health
Informatics 8, 7 (2018), 1368–1373.

The original version of the dataset is available under Creative Commons Attribution 4.0 International Public License,
from sources such as the original RITA repository linked, or

https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/#files-panel

## LLM - BoolQ

This dataset can be extracted largely as-is from the original source, but we have included our splits and shuffled versions.

https://github.com/google-research-datasets/boolean-questions

Per the dataset creators, BoolQ is released under the Creative Commons Share-Alike 3.0 license.

## Face - IMDB-Face

https://github.com/fwang91/IMDb-Face

This dataset is retreived using the author's script, producing a folder for "clean" faces, and within that a folder for each person's face images. Our program will read from that outer folder and traverse inwards.


# Running

## Pyhon environment
Each experiment has its own Python 3.10 python environment, with dependencies described using the pip freeze .txt files in each experiment's folder. The details of these are largely shared, but the exact dependency state from each experiment machine has been captured to be safe.

## Build command
(Replace face_query with the desired executable)
```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+fma,+sse,+sse2,+sse3,+sse4,+sse4,+ssse3,+aes,+avx,+avx2" cargo build --verbose --release --bin face_query

export LD_LIBRARY_PATH=$(find $(pwd)/target/release/build -name "out" -exec find {} -name "lib" \; | xargs -I {} echo -n {}:)${LD_LIBRARY_PATH}

mv target/release/build/face_query face_experiments/face_query
```

When inside the folder, all data will need to be extracted from the archive files (`tar -xzvf filename.tar.gz`), and the experiment will be run with a command like this (replacing the desired experiment's bash file as needed)

```sh
nohup time env RUST_BACKTRACE=full LD_LIBRARY_PATH=$LD_LIBRARY_PATH bash ./run_face_configs_std.sh > face_std.log.txt 2>&1 &
```

## LLM
You will need to copy the dataset file to the folder you are running the experiment from (this should be llm_experiments).
```sh
cd llm_experiments
cp llm_data_splits/llm_original_distribution/boolq_dataset_dev_seed_0.jsonl ./boolq_dataset_dev.jsonl

cp llm_data_splits/llm_original_distribution/llm_viable_model_result_dicts_seed_0.json ./viable_model_result_dicts.json
```

You will also need to start the chroma server.

```sh
nohup /home/ubuntu/.pyenv/versions/3.10.15/bin/chroma run --path ./gpt_chroma_db/ > chroma_log.txt 2>&1 &
```
## ECG
```sh
cp ./ecg_original_distribution/seed_0_data.npy ./mini_time_series_generated_sequences_data.npy
cp ./ecg_original_distribution/seed_0_labels.npy ./mini_time_series_generated_sequences_labels.npy
```

## Face
```sh
cp ./face_original_distribution/IMDB_split_info_seed_0.json ./IMDB_split_info.json
```
For the dat itself, which is referenced within this descriptor file, you will need to have downloaded the relevant piece of the IMDB-Face dataset and stored the clean faces in IMDB_clean_face.



