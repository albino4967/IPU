# docker 사용 방법
현재 진행 내용은 모두 `POPLAR SDK 3.0`을 기준으로 합니다.   


## 1. Build docker image
필요한 프레임워크와 모델을 선택하여 해당 디렉토리 내에서 다음 명령어를 실행합니다.
```bash
docker build -t <도커 이미지 이름>:<태그> .
```

## 2. Create a docker container
```bash
gc-docker -- -d --rm -it --name <컨테이너 이름> -p <host-port>:<container-port> -v <dataset-path>:/dataset <도커 이미지 이름>:<태그>
```

## 3. Execute docker container shell

```bash
docker exec -it <컨테이너 이름> bash
```

## 4. Run training

| model                 | framework  | train command                                                                                                                                                                                                                                                  | 
|-----------------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| bert-large packed-128 | pytorch    | `python3 run_pretraining.py --config pretrain_large_128 --checkpoint-output-dir --packed data checkpoints/pretrain_large_128`                                                                                                                                  |
| bert-large packed-512 | pytorch    | `python3 run_pretraining.py --config pretrain_large_512 --checkpoint-output-dir checkpoints/pretrain_large_512 --pretrained-checkpoint checkpoints/pretrain_large_128/step_N/`                                                                                 |
| bert-large packed-512 | popart     | `python bert.py --config=./configs/podXX-XXX.json`                                                                                                                                                                                                             | 
| resnet                | tensorflow | ```bash run.sh <dataset-path>```                                                                                                                                                                                                                               | 
| resnet                | pytorch    | `python3 train.py --config <config name> --auto-loss-scaling`                                                                                                                                                                                                  |
| rnnt                  | popart     | `python3 transducer_train.py --model-conf-file configs/transducer-1023sp.yaml --model-dir /localdata/transducer_model_checkpoints --data-dir /localdata/datasets/LibriSpeech/ --enable-half-partials --enable-lstm-half-partials --enable-stochastic-rounding` | 

기본적인 학습 명령어이며 데이터셋 경로가 다르기 때문에 적절하게 변경하여 사용해야 합니다.   
config 파일은 그래프코어에서 제공하는 파일을 참고하여 사용합니다.