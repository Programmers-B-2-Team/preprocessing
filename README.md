# Team B-2 final project
## Virtual Try On : PF-AFN
- 필요 패키지 설치
- `pip install -r requirements.txt`



#### 1. Binary Masking

- 의류 이미지에 대한 Binary Masking을 진행한다. Image file만을 사용하여 만드는 경우, 현재 코드로는 옷의 White color에 대해 적절하게 작동하지 않아 추가적인 작업이 필요한 상태다. 이후 다른 알고리즘을 적용한 방식을 시도하여 업데이트 할 예정.

- 모델의 트레이닝 단계에서는 AI Hub 데이터만 사용하므로 학습은 문제없이 진행이 가능할 것으로 생각된다. 그러나 서비스 단계에서 사용자의 이미지 파일에 대한 전처리 파이프라인을 구현할 것을 고려하여 Image only 케이스의 경우 또한 작업이 필수적일 것으로 보인다.

    

- JSON file을 사용한 Binary masking(AI Hub Data)

```bash
python main.py binary-mask <Image directory path> --json-path <JSON directory path> --save-path <Results directory>
```

- Image only Binary masking(New Data)

```bash
python main.py binary-mask <Image directory path> --save-path <Results directory> --image-only
```

- Image directory path : 옷 이미지 파일이 저장된 디렉토리 경로 argument (required)

- –json-path 옵션: 옷의 JSON 파일이 저장된 디렉토리 경로

- –save-path 옵션: Binary masked 된 이미지를 저장할 디렉토리 경로. default값은 `./item_mask`

- –image-only 옵션: JSON file없이 이미지만으로 Binary masking image를 생성할 경우 추가

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

- 실행 예

- ```bash
    python main.py binary-mask ./Item-Image --json-path ./Item-Parse_f
    ```

    

#### 2. Segmentation Labeling

- 모델 이미지에 대한 Segmentation Labeling을 진행한다.

- 현재 사용 중인 label 종류는 21가지로, 이는 CP VTON+ 논문을 참고하였다. PF-AFN 논문에서는 12개의 label을 부여하였다. Training code 또한 이에 맞춰져 있으므로, AI Hub data를 사용해 학습을 진행할 경우 Training code의 일부를 수정해줘야 한다.


```bash
python main.py segmentation-label <Image directory path> <JSON directory path> --save-path <Results directory>
```

- Image directory path : 모델 이미지 파일이 저장된 디렉토리 경로 argument (required)

- JSON directory path: 모델 이미지 JSON 파일이 저장된 디렉토리 경로 argument (required)

- –save-path 옵션: segmentation 된 이미지를 저장할 디렉토리 경로. default값은 `./model_segmentation`

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

- 실행 예

- ```bash
    python main.py segmentation-label ./Model-Image ./Model-Parse_f
    ```

    

#### 3. Pose Map

- 모델의 포즈에 대한 좌표 데이터로, AI hub에서 제공하는 Model Pose Parsing 데이터를 사용한다. 논문의 학습 코드에서 좌표값을 담고 있는 매핑 값만 조절해주면 무리없이 작동할 것으로 보인다.

    

#### 4. DensePose Map

- PF-AFN 논문에서 dense correspondense를 보다 정확히 얻기 위해서 사용한다. 페이스북의 detectron framework를 사용하여 densepose map을 얻을 수 있다.(해당 논문에서는 facebook research팀이 개발한 영상 전처리 프레임워크인 detectron으로 영상처리 기능이 통합되기 전의 코드를 사용한 것으로 보인다.)

    

- detectron2를 사용하기 위해서는 다음과 같은 절차를 거쳐야 한다.

    - 1) Detectron2 project를 다음 [Github](https://github.com/facebookresearch/detectron2 )에서 clone한다.

    - 2. pytorch, python, cuda의 버전에 맞는 detectron 프레임워크를 설치해야 한다. 이하 링크를 통해 확인할 것.

        - https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

    - 3. GPU 환경이 갖춰져야 기능이 동작한다. 로컬 환경에서 이를 실행하기 어려울 경우, Colab을 통해서 실행할 수 있다. Colab에서 실행할 경우, 다음의 설치 Tutorial을 제공한다.(이를 통해 Colab환경에 실행에 필요한 패키지 및 2번 항목의 프레임워크 설치를 한다. 이하 문서의 Install detectron2 항목을 참고.)

        - https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=FsePPpwZSmqt

    - 4. detectron을 사용하기 전, model checkpoint를 작업환경에 설치해야 한다. Densepose의 경우 다음의 두 가지 모델의 체크 포인트를 제공한다.

        - https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/DENSEPOSE_IUV.md
        - https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/DENSEPOSE_CSE.md
        - 이번 프로젝트에서는 DensePose IUV 모델의 체크포인트를 사용하였다. [R_101_FPN_DL_s1x](https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml)를 설치하고 이를 프로젝트 작업환경으로 이동시킨다. 앞서 1번 항목에서 Clone한 detectron2 프로젝트에서, detectron2/projects/DensePose 디렉토리 내로 이동시킨다.

    - 5. 설치가 완료된 후, Densepose 기능을 사용하는 것은 다음 문서를 참고한다.

        - https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/GETTING_STARTED.md

            

- **Colab 환경 실행 코드**

    - 로컬 환경에서 GPU로 인해 실행에 제약이 생겨, Colab으로 실행하였다. 다음 링크를 통해 코드를 확인할 수 있다. 실제 사용 시에는 해당 코드 내에서 디렉토리 설정을 각자의 작업환경에 맞게 수정해주도록 한다.

    - https://colab.research.google.com/drive/1Vypw8tKTlkAK_rJ4SxJVTLocBWxLUg8b?usp=sharing

        

- **로컬 환경(GPU 탑재시) 실행 코드**

    - :construction: 실제 로컬 환경에서는 GPU의 부재로 인해 실행을 하지 못하므로 아래 코드가 정확히 작동하는지는 확인하지 못했다. 이를 참고할 것.

```bash
python main.py densepose-map <Detectron directory path> <Image directory path> --save-path <Results directory>
```

- Detectron directory path : git으로부터 clone한 detectron2 프로젝트 내의 Densepose 디렉토리 경로 argument (required)

- Image directory path: 모델 이미지 파일이 저장된 디렉토리 경로 argument (required)

- –save-path 옵션: densepose numpy file을 저장할 디렉토리 경로. default값은 `./model_densepose`

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

- 실행 예

- ```bash
    python main.py segmentation-label ./detectron2/projects/DensePose ./Model-Image
    ```

    



