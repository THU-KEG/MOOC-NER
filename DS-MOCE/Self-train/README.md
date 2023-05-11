# Step 3: Discipline-embedding models with self-training

## Task
Using distant labels to self-train PLMs to do sequence labeling task.

## Method

### Model
External academic field features are integrated into the embedding, enhancing model discipline-aware capability.

### Training Strategy
DS-MOCE(co): co-training to denoise using two teacher-student networks.
DS-MOCE(PUL): positive-unlabeled learning (PUL) to handle the incomplete challenge.

## Requires

**env**: 
- python==3.7.4
- pytorch==1.6.0
- [huggingface transformers](https://github.com/huggingface/transformers)
- numpy
- tqdm

**input**:
or dataset directory: (copy from share_data) 
- a discipline-aware Dictionary: `./dataset/output_embeddings.csv`
- `mooc_train.json` from distant supervision 
- `mooc_test.json` and `mooc_dev.json` (same with the test set) to evaluate





## Run

### Usage

For DS-MOCE-co on CUDA=4:
```
bash ./DS-MOCE-co/run_co.sh 4 mooc
```

For DS-MOCE-PUL on CUDA=4:
```
bash ./DS-MOCE-PUL/run_pul.sh 4 mooc
```