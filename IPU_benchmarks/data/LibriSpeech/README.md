## LibriSppech 데이터셋 구성

약 1000시간 분량의 16kHz 영어 음성 다중 화자 데이터셋인 LibriSpeech 데이터셋을 구성합니다.   
참고 : http://www.openslr.org/12.
   
LibriSppech 데이터셋을 구성하는데 120GB가 필요합니다. 충분한 공간을 확보한 후 진행합니다.   
현재 쉘 스크립트는 mlcommons에서 Graphcore가 제공하는 스크립트와 동일합니다.   
참고 : https://github.com/mlcommons/training_results_v2.0/tree/main/Graphcore/benchmarks/rnnt/implementations/popart/training#download-and-preprocess-the-librispeech-dataset   

데이터셋 구성시 데이터 경로는 `/localdata/datasets`를 가정하여 다음 순서로 진행합니다.
### 1. 다운로드
```
bash scripts/download_librispeech.sh /localdata/datasets
```

### 2. 데이터 전처리
```
bash scripts/preprocess_librispeech.sh /localdata/datasets
```

### 3. sentece pieces 적용
```
bash scripts/create_sentencepieces.sh /localdata/datasets
```