import json


# crop & resized image에 맞게 원본 이미지의 포즈 좌표값을 이동
def new_posemap(pose_path, pose_json, pose_save, bbox, resize_ratio):
    json_data = json.load(open(f'{pose_path}/{pose_json}'))
    pts = json_data['landmarks']
    new_points = []
    for x, y, _ in [pts[i:i + 3] for i in range(0, len(pts), 3)]:
        new_x = (x - bbox[0]) * resize_ratio
        new_y = (y - bbox[1]) * resize_ratio
        new_points += [new_x, new_y]
    json_data['landmarks'] = new_points
    with open(f'{pose_save}/{pose_json}', 'w') as f:
        json.dump(json_data, f)
