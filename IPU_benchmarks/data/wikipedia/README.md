# BERT-Large 학습 데이터 구성 방법

그래프코어에서는 학습 속도 향상을 위한 packed 알고리즘을 적용한 데이터셋을 사용합니다.   
자세한 흐름은 https://github.com/graphcore/examples/tree/master/nlp/bert/popart/bert_data 를 참고하세요.

## 데이터셋 다운로드 및 전처리
`download-wikidump.sh` wikipedia dump를 다운로드 받습니다. wikidump.xml 파일을 다운받을 수 있습니다.
`preprocess/preprocess.sh` wiki dump에서 학습 데이터를 추출한 후 전처리를 진행합니다. sequence length (128, 384, 512) 에 따라 다른 종류의 데이터를 확인할 수 있습니다.

## packed data 생성
위 데이터를 학습으로 사용할 수 있지만 packed dataset을 구성하면 더 빠른 성능을 확인할 수 있습니다.   
이때 기존 데이터셋을 구성할 때 사용한 옵션을 동일하게 주어야 합니다. 해당 내용은 그래프코어 제공 git의  config 옵션을 통해 확인할 수 있습니다.
아래 표는 몇개의 sample을 제시합니다.

|  | framework | sequence-length | mask-tokens | duplication-factor | max-sequence-per-pack |
| --- | --- | --- | --- | --- | --- |
| bert-large packed sl128 | pytorch | 128 | 20 | 1 | 3 |
| bert-large packed sl512 | pytorch | 512 | 76 | 1 | 3 |
| bert-large packed sl512 | popart | 512 | 76 | 3 | 3 |

`packing` 디렉토리는 https://github.com/graphcore/examples/tree/master/nlp/bert/pytorch/data/packing 내용과 동일합니다.
위 내용을 참고하여 다음과 같은 명령어를 수행하면 packed dataset 구성이 가능합니다.
```
python3 -m bert_data.pack_pretraining_data --input-glob="preprocessed_target_folder/wiki_*" --output-dir="packed_pretraining_data" --max-sequences-per-pack=3 --mlm-tokens=76 --max-sequence-length=512 --unpacked-dataset-duplication-factor=6
```
