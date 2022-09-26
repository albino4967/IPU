# poptorch 설치 및 사용

poptorch 사용을 위한 설치 방법을 확인합니다.   
먼저 그래프코어에서 제공하는 poplar SDK를 다운로드하여 설치합니다.
*최신 버전 사용을 권장합니다.*

|poptorch|torch|torchvision|python|
|------|---|---|---|
|2.6|1.10.0|0.11.1|>=3.6|
|2.5|1.10.0|0.11.1|>=3.6|
|2.4|1.10.0|0.11.1|>=3.6|
|2.3|1.9.0|0.10.0|>=3.6|
|2.2|1.9.0|0.10.0|>=3.6|
|2.1|1.7.1|0.8.2|>=3.6|
|2.0|1.7.1|0.8.2|>=3.6|
|1.4|1.6.0|0.7.0|>=3.6|


## python virtual environment 사용

IPU를 사용할 때 추천하는 방법입니다.   
'''
virtualenv -p python3 poptorch_test      
source poptorch_test/bin/activate   
pip install -U pip   
pip install <sdk_path>/poptorch_x.x.x.whl   
'''
## 환경변수 세팅


'''
source poptorch_test/bin/activate   
#2.6 부터는 poplar installation만 진행하면 됩니다.   
source <path to poplar installation>/enable.sh   
source <path to popart installation>/enable.sh   
'''


## 설치 확인

다음 커맨드를 통하여 poptorch 설치 여부를 확인합니다.   
작성한 예제 모델이 IPU에 컴파일이 된 후 설치가 잘 이루어졌다면 성공이 출력됩니다.
<code>
python validation.py
</code>
