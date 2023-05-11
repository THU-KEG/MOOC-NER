# README.md

download [here](https://cloud.tsinghua.edu.cn/f/aae5acc586904015afd6/)

## Resource
- dictionary: `all_fields_concepts.csv`
- MOOC video subtitles for model training: `tmp_train`
- expert-labeled test set: `tmp_test` / `test.json`
- intermediate result: (can be ignored for developing your own strategy)
  -`output_embeddings.csv` : a discipline-aware Dictionary after Step1 of DS-MOCE
  -`train.json`: the distant labels after Step2 of DS-MOCE

### Dictionary:
`all_fields_concepts.csv` is an expert-checked dictionary with over 100k course concepts from [CNCTST]{http://www.cnterm.cn/} spanning 20 disciplines.

each row contains: concept names, related academic disciplines
e.g. \
```
《三国史记》,世界历史 
活度,化学,机械工程,材料科学技术,物理学,电气工程 
```

and 20 disciplines in Chinese are:
```
心理学
教育学
语言学
世界历史
数学
物理学
化学
力学
机械工程
材料科学技术
电气工程
计算机科学技术
建筑学
船舶工程
航天科学技术
航空科学技术
农学
医学
免疫学
管理科学技术
```

For more information, see Appendix of our paper.


### MOOC video subtitles:
`tmp_train` directory contains a subtitles corpus from $315$ courses with $167,496$ unlabeled character sequences on average per course.
For more information, see Appendix of our paper.

Each course json file contains:
- `id`: course unique id
- `name`: course name
- `fields`: course related fields, a list.
- `videos`: course video resources, a list.
  - for each video:
    - `id`: video unique id
    - `titles`: video captions
    - `ccid`: video subtitle unique id
    - `text`: a string list of cideo subtitles
    - `str_words`: a char list of text joined with '，'
    - `tags`: distant labels that can be updated

Note:`tmp_test` json files' attributes are same with above, with expert-labeled `tags` as ground truth.

### Expert-labeled Test Set

The test set contains $522$ expert-annotated sentences from $17$ courses with $15,375$ discipline-related course concepts.
For more information, see Appendix of our paper.
- `tmp_test` for debug
- `test.json` for evaluation
  - only contains `str_words` and `tags` for each instance and length of sequences is set 512.

#### Evaluation of DS-NER Methods:


| Method | P | R | F1 |
| - | - | - | - |
|**Dic-Matching** |||
DM | 12.50 | 25.84 | 16.85 
DM(AD-human) | 22.95 | 17.38 | 19.78 
DM(AD-GLM) | 34.59 | 15.40 | 21.31 
|**Distant-Sup** |||
SCDL | 34.59 | 21.16 | 26.26 
RoSTER | 35.40 | 26.70 | 30.40 
BOND | 32.37 | 44.78 | 37.58 
|**Our DS-MOCE**|||
DS-MOCE(co) | **81.93** | 30.82 | **44.79**
DS-MOCE(PUL) | 34.53 | **49.34** | 40.62 
|**Sup.**|||
FLAT | 56.08 | 57.17 | 56.62 


## DS-MOCE:
- step1: Disicipline-aware Dictionary Empowerment
  - input: `all_fields_concepts.csv`
  - output: `output_embeddings.csv`
- step2: Distant Supervision Refinement
  - input:
    - dictionary: `output_embeddings.csv`  
    - original course corpus:`tmp_train`
  - output:
    - `train.json` 
- step3: Discipline-embedding models with self-training
  - `train.json` for model training
  - `test.json` for model evaluation