import mathdef get_center_of_bbox(bbox):    x1, y1, x2, y2 = bbox    center_x = int((x1 + x2) / 2)    center_y = int((y1 + y2) / 2)    return (center_x, center_y)def measure_distance(p1, p2):    # Hypotenuse of a Right Triangle Formula    # return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))def get_foot_position(bbox):    x1, y1, x2, y2 = bbox    # we find the bbox centroid. we then get the maximum 'y' value.    return (int((x1 + x2) / 2), y2)def get_closest_keypoint_index(point, keypoints, keypoint_indices):    # point: (x,y) coordinate in pixels.    # keypoints: [x1, x2, ..., x28] all kps in pixels.    # keypoint_indices: [0, 2, 12, 13] indices we want to compare against..    closest_distance = float("inf")    key_point_ind = keypoint_indices[0]    for idx in keypoint_indices:        keypoint = keypoints[idx * 2], keypoints[idx * 2 + 1]        distance = abs(point[1] - keypoint[1])        if distance < closest_distance:            closest_distance = distance            key_point_ind = idx    return key_point_inddef measure_xy_distance(p1, p2):    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])def get_height_of_bbox(bbox):    return bbox[3] - bbox[1]