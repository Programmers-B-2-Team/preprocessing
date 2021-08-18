# Team B-2 final project
## Virtual Try On : PF-AFN
### Preprocessing Code

- 필요 패키지 설치
- `pip install -r requirements.txt`

1. Binary Masking

    - 의류 이미지에 대한 Binary Masking을 진행한다. Image file만을 사용하여 만드는 경우, 현재 코드로는 옷의 White color에 대해 적절하게 작동하지 않아 추가적인 작업이 필요한 상태다. 이후 다른 알고리즘을 적용한 방식을 시도하여 업데이트 할 예정.

    - 모델의 트레이닝 단계에서는 AI Hub 데이터만 사용하므로 학습은 문제없이 진행이 가능할 것으로 생각된다. 그러나 서비스 단계에서 사용자의 이미지 파일에 대한 전처리 파이프라인을 구현할 것을 고려하여 Image only 케이스의 경우 또한 작업이 필수적일 것으로 보인다.

        

    - JSON file을 사용한 Binary masking(AI Hub Data)

    ```bash
    python main.py binary-mask <Image directory Path> --json-path <JSON directory Path> --save-path <Results directory>
    ```

    - Image only Binary masking(New Data)

    ```bash
    python main.py binary-mask <Image directory Path> --save-path <Results directory> --image-only
    ```

    - Image directory Path : 옷 이미지 파일이 저장된 디렉토리 경로 argument

    - –json-path 옵션: 옷의 JSON 파일이 저장된 디렉토리 경로

    - –save-path 옵션: Binary masked 된 이미지를 저장할 디렉토리 경로. default값은 `./masked`

    - –image-only 옵션: JSON file없이 이미지만으로 Binary masking image를 생성할 경우 추가

    - 각 디렉토리 경로는 상대경로로 입력할 수 있다. 기준이 되는 것은 main.py의 위치.

    - 실행 예

    - ```bash
        python main.py binary-mask ./Item-Image --json-path ./Item-Parse_f
        ```

        

