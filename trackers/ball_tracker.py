import cv2
import numpy as np
import pandas as pd
import pickle
from ultralytics import YOLO

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
from scipy.signal import find_peaks


class BallTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_frame(self, frame: np.ndarray):
        # Retrieving the YOLO model predictions.
        # Not tracking since the model only detects 1 ob
        results = self.model.predict(source=frame, conf=0.15)[0]

        # Initializing an empty ball_dict dictionary to store predictions.
        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]  # bounding box coordinates (xyxy)
            ball_dict[1] = result  # "tennis_ball" is hard coded to ID of 1.

        return ball_dict

    def detect_frames(
        self,
        frames: np.ndarray,
        read_from_stub: bool = False,
        stub_path: str = None,
    ):
        # Initializing an empty array to store model predictions for each frame.
        ball_detections = []

        # If ball_detections has been pre-computed and saved, we use those.
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # If not, iterate through each frame and get the ball specific predictions.
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Saving the ball_detections
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def interpolate_ball_positions(self, ball_positions):
        # Retrieving all ball predictions and storing them in an array.
        ball_positions = [x.get(1, []) for x in ball_positions]  # "tennis_ball" ID = 1.
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Fill NaN values using an interpolation method
        df_ball_positions = df_ball_positions.interpolate()

        # Fill NaN values using next valid observation.
        df_ball_positions = df_ball_positions.bfill()  # for the first observation (if None).

        # Convert dataframe back to array.
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        # returing the interpolated ball_positions
        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_pos = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_pos["ball_hit"] = 0

        df_ball_pos["mid_y"] = (df_ball_pos["y1"] + df_ball_pos["y2"]) / 2
        df_ball_pos["mid_y_rolling_mean"] = df_ball_pos["mid_y"].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_pos["delta_y"] = df_ball_pos["mid_y_rolling_mean"].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_pos) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_pos["delta_y"].iloc[i] > 0 and df_ball_pos["delta_y"].iloc[i + 1] < 0
            positive_position_change = df_ball_pos["delta_y"].iloc[i] < 0 and df_ball_pos["delta_y"].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = (
                        df_ball_pos["delta_y"].iloc[i] > 0 and df_ball_pos["delta_y"].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                        df_ball_pos["delta_y"].iloc[i] < 0 and df_ball_pos["delta_y"].iloc[change_frame] > 0
                    )

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_pos["ball_hit"].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_pos[df_ball_pos["ball_hit"] == 1].index.tolist()

        return frame_nums_with_ball_hits

    def get_ball_shot_frames_V2(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_pos = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_pos["ball_hit"] = 0

        df_ball_pos["mid_y"] = (df_ball_pos["y1"] + df_ball_pos["y2"]) / 2
        df_ball_pos["mid_y_rolling_mean"] = (
            df_ball_pos["mid_y"]
            .rolling(
                window=5,
                min_periods=1,
                center=False,
            )
            .mean()
        )

        smoothed_mid_y = df_ball_pos["mid_y_rolling_mean"].values
        inverted_mid_y = -smoothed_mid_y  # Invert if y increases downward

        # Detect peaks (ball moving upward)
        peaks, _ = find_peaks(inverted_mid_y, distance=10, prominence=5)

        # Detect troughs (ball moving downward)
        troughs, _ = find_peaks(smoothed_mid_y, distance=10, prominence=5)

        peaks_and_troughs = np.concatenate((peaks, troughs))
        peaks_and_troughs.sort()

        # returns the frames where the signal peaks and drops.
        return peaks_and_troughs  # frame_nums_with_ball_hits

    def draw_bboxes(self, video_frames, ball_detections):
        # Initializing a list of empty output annotated frames
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Ball ID: {track_id}",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2,
                )
            output_video_frames.append(frame)

        return output_video_frames
