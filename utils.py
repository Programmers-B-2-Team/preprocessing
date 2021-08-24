# directory 내의 파일 이름을 리스트로 얻는다.
import os
import shutil
from os import walk


def get_file_list(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
    return files


def move_listed_files(old_path,new_path, lst):
    os.makedirs(new_path, exist_ok=True)
    for i in lst:
        shutil.copy(old_path+ "/" + i, new_path)


# 모델 파일명의 마지막 숫자는 카메라를 기준으로 모델 포즈의 각도를 의미하는 것 (데이터 명세서에 따르면)
# 정면 사진만 활용할 것이기 때문에 "000.jpg"로 끝나는 파일이 아니면, 모두 삭제
# 특정 스트링이 포함된 리스트아이템을 모아 리스트를 만든다.
def new_list_certain_str(lst, string):
    new_lst = [item for item in lst if string in item]
    return new_lst
