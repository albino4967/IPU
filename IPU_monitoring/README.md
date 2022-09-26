# IPU 모니터링

IPU는 GPU와 다르게 모델을 각각의 노드에 미리 세팅한 후 학습을 진행합니다.   
때문에 GPU에서 사용하는 `nvidia-smi`와 같은 `gc-monitor`로 현재 사용가능한 IPU와 사용되고 있는 IPU를 확인 할 수 있지만 미리 작업을 올리고 시작하기 때문에 현재 사용량은 출력되지 않습니다.

## popvision

IPU는 메모리 사용량을 확인할 수 있는 툴을 제공합니다.   
popvision 다운로드 링크 : [다운로드][https://www.graphcore.ai/developer/popvision-tools]   

## 사용법

popvision을 사용하기 위해서는 예제 코드를 실행 할 때 옵션을 주어 프로파일링을 수행합니다.
```
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./IPU_profile"}' python ipu_profile.py
```
수행하면 IPU_profile 디렉토리가 생성됩니다. popvision을 실행하여 서버에 연결한 후 IPU_profile 디렉토리 내에서 `inference` 디렉토리를 확인한 후 Open 수행하면 메모리 사용량을 확인할 수 있습니다.   
만약 training 모델에 수행하였다면 `training`이 생성되며 두개가 동시에 수행되면 모두 생성되는 것을 확인할 수 있습니다.