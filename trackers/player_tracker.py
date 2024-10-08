import sys

sys.path.append("../")

from ultralytics import YOLO
import numpy as np
import cv2
import pickle
from utils import get_center_of_bbox, measure_distance


class PlayerTracker:
    def __init__(self, model_path: str):
        # Initializing the pretrained model using the `model_path` ckpt
        self.model = YOLO(model_path)

    def detect_frame(self, frame: np.ndarray):
        results = self.model.track(source=frame, persist=True)[0]
        id_name_dict = results.names  # {id:object}. Example: {0:'person'}

        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])  # track ID
            result = box.xyxy.tolist()[0]  # bounding box coordinates
            object_cls_id = box.cls.tolist()[0]  # class category ID
            object_cls_name = id_name_dict[object_cls_id]  # class name

            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def detect_frames(
        self,
        frames: np.ndarray,
        read_from_stub: bool = False,
        stub_path: str = None,
    ):
        # Initializing an empty array to store model predictions for each frame.
        player_detections = []

        # If player_detections has been pre-computed and saved, we use those.
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            return player_detections

        # If not, iterate through each frame and get the player specific predictions.
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # Saving the player_detections to avoid computing the results again.
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    # Selects the 2 whose BBOX centers are closest to any of the court keypoints
    def choose_players(self, court_keypoints, player_dict):
        distances = []  # initializing an empty array of distances

        # iterating through all the detected players in a given frame.
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)  # obtaining the center point of BBOX.

            min_distance = float("inf")

            # Calculating the distance from the player's center to every court keypoint.
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            # Storing only the min_distance for each player.
            distances.append((track_id, min_distance))

        # sorrt the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks, i.e., track_ids
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players  # returns 2 track_ids with minimum distance from the court kps

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # using the first frame to choose the players to track.
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(
            court_keypoints,
            player_detections_first_frame,
        )  # 'chosen_players' is a list of player IDs, e.g., [1, 3]

        # filter detections in all frames to include only the chosen players
        filtered_player_detections = []
        for player_dict in player_detections:
            # Keep only the detections of the chosen players in the current frame
            filtered_player_dict = {
                track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player
            }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    # Annotate the video_frames with player_detections
    def draw_bboxes(self, video_frames, player_detections):
        # Initializing a list of empty output annotated frames
        output_video_frames = []
        # Looping over each frame and its corresponding PlayerTracker prediction
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                # Draw Bounding Boxes
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2,
                )
            output_video_frames.append(frame)

        return output_video_frames


"""
YOLOV9 Model Classes
{0: 'person',
 1: 'bicycle',
 2: 'car',
 ...
 78: 'hair drier',
 79: 'toothbrush'}
"""
