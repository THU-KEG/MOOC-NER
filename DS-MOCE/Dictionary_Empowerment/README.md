# Step 1 : Discipline-aware Dictionary Empowerment

## Task
Given each concept in a Dictionary, output it's related discipline and distribution across all domains.

## Method

- prompt-based learning;
- Hearst Pattern templates;
- GLM (General Language Model); \
    We apply [GLM](https://github.com/THUDM/GLM) to our classification task in zero-shot setting without fine-tune. \
    We use GLM-Large-Chinese checkpoints of GLM. \
    For downloading and more information, see `README.md` of GLM. 

## Requires

**env**: \
    - same `requirements.txt` with GLM repo. \
**input**: \
    - a Dictionary \
**output**:\
    - distribution across all field 

## Run

### Usage

Put `run.sh` and `generate_field.py` in GLM submodule, then:

```
CUDA_VUSUBLE_DEVICES=1 bash run.sh
```
### Arguments:
you can change the following arguments in `run.sh`.

- concept_dictionary: 
    filename of dictionary : `./data/all_fields_concepts.csv`(copy from share_data)
- prompt_template:
    prompt template which contains `[concept]` and `[MASK]` slots for field clozing \
    e.g. `[MASK]领域中有很多重要概念，其中[concept]是本节课的重点。`
- short_setting:
    you can modify the number and name of fieldlists in `./GLM/generate_fields.py`
- concept_output_embeddings:
    output which is required by the following two steps.\
    default : `./data/output_embeddings.csv `
