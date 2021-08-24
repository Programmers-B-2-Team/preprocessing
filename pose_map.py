import json


def new_posemap(pose_path, pose_json, pose_save, bbox):
    json_data = json.load(open(f'{pose_path}/{pose_json}'))
    pts = json_data['landmarks']
    width, height = json_data['image_size']['width'], json_data['image_size']['height']
    new_points = []
    for x, y, _ in [pts[i:i+3] for i in range(0, len(pts), 3)]:
        new_x = (x / width) * (bbox[2] - bbox[0])
        new_y = (y / height) * (bbox[3] - bbox[1])
        new_points += [new_x, new_y]
    json_data['landmarks'] = new_points
    with open(f'{pose_save}/{pose_json}', 'w') as f:
        json.dump(json_data, f)
