

# Team B-2 final project

## Virtual Try On : PF-AFN
- 필요 패키지 설치
- `pip install -r requirements.txt`



#### 1. Binary Masking

- 의류/모델 이미지에 대한 Binary Masking을 진행한다. Image file만을 사용하여 만드는 경우, Grabcut을 이용한 코드로 binary masking을 실행한다. 단 이 경우, 흰 배경 + 밝은 옷의 조합에서 부정확한 masking 결과가 나타나기 때문에, 데이터셋이 어느정도 조건에 부합하게 필터링되어있지 않다면 U Net을 사용한 method를 실행시키는 것이 더 정확한 결과를 기대할 수 있다.

      

- JSON file을 사용한 Binary masking(AI Hub Data)

```bash
python main.py binary-mask <Image directory path> <Results directory> --json-path <JSON directory path>
```

- Image only Binary masking(New Data)

```bash
python main.py binary-mask <Image directory path> <Results directory> --image-only
```

- Image directory path : 옷/모델 이미지 파일이 저장된 디렉토리 경로 argument (required)

- Results directory: Binary masked 된 이미지를 저장할 디렉토리 경로 argument (required)

- –json-path 옵션: 옷/모델의 JSON 파일이 저장된 디렉토리 경로

- –image-only 옵션: JSON file없이 이미지만으로 Binary masking image를 생성할 경우 추가

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

    

- 실행 예

- ```bash
    python main.py unet-mask ./Item-Image ./model_masked --json-path ./Item-Parse_f
    ```



#### 1-2. Binary Masking with U Net

- 의류/모델 이미지에 대한 Binary Masking을 진행한다. Image만을 사용하며, U Net을 활용하여 마스킹을 실행한다. 이 경우 딥러닝 추론과정을 거치기 때문에 위 1번 방식보다는 조금 느린 성능을 보이나, 대신 옷과 배경의 색과 상관없이 더욱 정확한 결과를 도출할 수 있다.
- 실행 전 반드시 다음의 패키지들을 설치해야만 한다. 각각 Image Augument와 Image segmentation model을 불러오는 기능을 수행한다.
    - albumentations==1.0.3
    - segmentation-models-pytorch==0.2.0

```bash
python main.py unet-mask <Image directory path> <Results directory>
```

- Image directory path : 옷/모델 이미지 파일이 저장된 디렉토리 경로 argument (required)

- Results directory: Binary masked 된 이미지를 저장할 디렉토리 경로 argument (required)

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치

    

- 실행 예

- ```bash
    python main.py binary-mask ./Item-Image ./model_masked
    ```






#### 2. Segmentation Labeling

- 모델 이미지에 대한 Segmentation Labeling을 진행한다.
- 현재 사용 중인 label 종류는 21가지로, 이는 CP VTON+ 논문을 참고하였다. PF-AFN 논문에서는 12개의 label을 부여하였다. Training code 또한 이에 맞춰져 있으므로, AI Hub data를 사용해 학습을 진행할 경우 Training code의 일부를 수정해줘야 한다.
- 3 channels을 가진 RGB image가 아니라 2차원의 Indexed color 기반의 png image를 생성하도록 한다.


```bash
python main.py segmentation-label <JSON directory path> --save-path <Results directory>
```

- JSON directory path: 모델 이미지 JSON 파일이 저장된 디렉토리 경로 argument (required)

- –save-path 옵션: segmentation 된 이미지를 저장할 디렉토리 경로. default값은 `./model_segmentation`

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

    

- 실행 예

- ```bash
    python main.py segmentation-label ./Model-Parse_f
    ```

    

#### 3. Resizing

- 해당 파트에서는 PF-AFN 논문에서 사용한 이미지의 비율에 맞게 리사이즈하는 작업을 진행한다.
    - 옵션을 통해서 의류 Item image / Item Mask images에만 Resizing 할 지, 
    - 혹은 Model image / Model mask / Segmentation / Pose Resizing 및 JSON data 생성을 진행할 지 결정한다.

- AI hub에서 Pose JSON data를 제공하기 때문에, PF-AFN 논문의 학습코드의 일부 매핑을 변경해주고 좌표값에 필요없는 일부 데이터를 삭제해주는 것으로 간단히 처리할 수 있다. 그러나 Resizing이 이미지 데이터에 추가되는 경우, 이에 대한 Pose map의 좌표값을 변경될 이미지 사이즈에 맞게 상대적 이동이 필요해진다.
    -  AI Hub에서 제공하는 Pose data의 경우, Pose의 point 수가 17개로 설정되어 있으나, 논문에서 사용하는 Pose point의 수는 17개 이상이 존재하기 때문에, 완벽하게 대응할 수 있을지 확인해 볼 필요가 있다.


```bash
python main.py resizing <Model image directory path> <Model mask directory path> --image-save <Results directory> --mask-save <Results directory>
--seg-path <Segmentation directory path> --pose-path <Pose JSON directory path> --seg-save <Results directory> --pose-save <Results directory> --is-item
```

- Model image directory path : 모델 이미지 파일이 저장된 디렉토리 경로 argument (required)

- Model mask directory path: 모델 이미지 binary mask 파일이 저장된 디렉토리 경로 argument (required)

- –image-path 옵션: segmentation 된 이미지를 저장할 디렉토리 경로. default값은 `./resize_image`

- –mask-path 옵션: segmentation 된 이미지를 저장할 디렉토리 경로. default값은 `./resize_mask`

    

- –seg-path : 모델 Segmentation 이미지 파일이 저장된 디렉토리 경로

- –pose-path: 모델 Pose JSON 파일이 저장된 디렉토리 경로

- –seg-save 옵션: segmentation 된 이미지를 저장할 디렉토리 경로. default값은 `./resize_segmentation`

- –pose-save 옵션: 새롭게 조정된 Pose JSON 데이터를 저장할 디렉토리 경로. default값은 `./resize_pose`

- –is-item 옵션: Item image만을 resize할 경우에는 –is-item을 추가하고, 그 외의 경우는 해당 옵션을 추가하지 않는다. 

- 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

    

- 실행 예

    - Item only

    ```bash
    python main.py resizing ./Item-Image ./Item-Mask --is-item
    ```

    - Model, Seg, Pose

    ```bash
    python main.py resizing ./Model-Image ./Model-Mask --seg-path ./Segmentation --pose-path ./Model-Pose
    ```




#### 4. DensePose Map

- PF-AFN 논문에서 dense correspondense를 보다 정확히 얻기 위해서 사용한다. 페이스북의 detectron framework를 사용하여 densepose map을 얻을 수 있다.(해당 논문에서는 facebook research팀이 개발한 영상 전처리 프레임워크인 detectron으로 영상처리 기능이 통합되기 전의 코드를 사용한 것으로 보인다.)

- 만약 이미지 데이터에 Resizing이 들어갈 경우, Resized 된 모델 이미지 셋을 사용해야 한다.

    

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

    



