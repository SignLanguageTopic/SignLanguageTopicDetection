# Sign Language Topic Detection

## Set up the environment

Clone this repository to your machine:
```bash
export FAIRSEQ_ROOT=... # Set this to the directory where you want to clone fairseq
export SL_DIR=${FAIRSEQ_ROOT}/examples/SL_topic_detection

git clone https://github.com/SignLanguageTopic/SignLanguageTopicDetection.git ${FAIRSEQ_ROOT}
```

Create the environment and activate it:
```bash
conda env create -f ${SL_DIR}/environment.yml && \
conda activate SLTopicDetection
```

Install fairseq:
```bash
pip install --editable ${FAIRSEQ_ROOT}
```

Define the root folder of [How2Sign](https://how2sign.github.io) as:
```bash
export H2S_ROOT=../../../data
```

The data in `H2S_ROOT` should be organized as follows:
```bash
${H2S_ROOT}
├── keypoints/
├── mediapipe_keypoints/
├── rotational/
├── mediapipe_rotational/
├── i3d
└── text
```

## Train Model

```bash

H2S_ROOT=...

FAIRSEQ_ROOT=./

FEATS_TYPE=keypoints

SP_MODEL=${H2S_ROOT}/text/spm_unigram8000_en.model

MODEL_TYPE=transformerCLS

CONFIG_NAME=baseline_${MODEL_TYPE}_${FEATS_TYPE}

NUM_EXP=1

echo $(pwd)

SEED=$NUM_EXP fairseq-hydra-train \
    +task.data=${H2S_ROOT}/${FEATS_TYPE} \
    +task.dict_path=${H2S_ROOT}/categoryName_categoryID.csv \
    +task.feats_type=$FEATS_TYPE \
    checkpoint.save_dir=../../../final_models/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP} \
    bpe.sentencepiece_model=${SP_MODEL} \
    --config-dir ${FAIRSEQ_ROOT}/config \
    --config-name ${CONFIG_NAME}

```


## Run Inference


```bash

FEATS_TYPE=text

H2S_ROOT=...

FAIRSEQ_ROOT=../../..

MODEL_TYPE=lstm

CONFIG_NAME=inference_${MODEL_TYPE}

SP_MODEL=${H2S_ROOT}/text/spm_unigram8000_en.model

OUTPUTS_DIR=${FAIRSEQ_ROOT}/outputs
mkdir -p $OUTPUTS_DIR

echo $(pwd)


MODEL_PATH=../../../final_models/${MODEL_TYPE}_${FEATS_TYPE}_${NUM_EXP}/checkpoint_best.pt
for DATASET_SPLIT in val test train
do
    echo '*************************************************'
    echo Starting experiment $NUM_EXP, $DATASET_SPLIT split, $FEATS_TYPE features
    echo '*************************************************'
    DATA=$H2S_ROOT \
    DICT_PATH=${H2S_ROOT}/categoryName_categoryID.csv \
    MODEL_PATH=$MODEL_PATH \
    CONFIG_NAME=${CONFIG_NAME} \
    SP_MODEL=${SP_MODEL} \
    DATASET_SPLIT=$DATASET_SPLIT \
    OUTPUTS_FILE=${OUTPUTS_DIR}/${CONFIG_NAME}_${FEATS_TYPE}_${DATASET_SPLIT}.pt \
    FEATS_TYPE=$FEATS_TYPE \
    MODEL_TYPE=${MODEL_TYPE} \
    python infer.py
    echo '*************************************************'
    echo Finishing experiment $NUM_EXP, $DATASET_SPLIT split
    echo '*************************************************'
    echo
done


```
