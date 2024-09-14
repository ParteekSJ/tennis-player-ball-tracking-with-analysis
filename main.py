import cv2
from copy import deepcopy
import pandas as pd

from config import get_cfg
from court_keypoints_detector import CourtKeypointsDetector
from mini_court import MiniCourt
from trackers import PlayerTracker, BallTracker

from utils import (
    read_video,
    save_video,
    measure_distance,
    convert_pixel_distance_to_meters,
    draw_player_stats,
)


def main():
    # Retrieving the config
    cfg = get_cfg()

    # Retrieve individual frames from the video
    video_frames = read_video(video_path=cfg.DATASET.SAMPLE_VIDEO_DIR)

    # Intialize the PlayerTracker
    player_tracker = PlayerTracker(model_path=cfg.MODEL.YOLOV9_CKPT_DIR)

    # Initialize the BallTracker
    ball_tracker = BallTracker(model_path=cfg.MODEL.TENNIS_BALL_MODEL_DIR)

    # Detect players in the given frames.
    player_detections = player_tracker.detect_frames(
        frames=video_frames,
        read_from_stub=True,
        stub_path="./tracker_stubs/player_detections.pkl",
    )

    # Detect tennis ball in the given frames.
    ball_detections = ball_tracker.detect_frames(
        frames=video_frames,
        read_from_stub=True,
        stub_path="./tracker_stubs/ball_detections.pkl",
    )

    # Applying LINEAR INTERPOLATION to model's ball predictions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Initializing the Court Keypoints Detector
    court_kps_detector = CourtKeypointsDetector(model_path=cfg.MODEL.COURT_KPS_MODEL_DIR)
    # Detecting Court Keypoints ONLY on the first frame
    court_kps = court_kps_detector.predict(video_frames[0])

    # Filtering out player_detections based on DISTANCE between keypoints and players.
    player_detections = player_tracker.choose_and_filter_players(
        court_kps,
        player_detections,
    )

    p1, p2 = player_detections[0].items()

    # Add the Minicourt onto the frame
    mini_court = MiniCourt(frame=video_frames[0])

    # Detect Ball Shot Frames
    # ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    ball_shot_frames = ball_tracker.get_ball_shot_frames_V2(ball_detections)

    # Convert the positions to mini-court positions
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections,
            ball_detections,
            court_kps,
        )
    )

    player_stats_data = [
        {
            "frame_num": 0,
            "player_1_number_of_shots": 0,
            "player_1_total_shot_speed": 0,
            "player_1_last_shot_speed": 0,
            "player_1_total_player_speed": 0,
            "player_1_last_player_speed": 0,
            "player_2_number_of_shots": 0,
            "player_2_total_shot_speed": 0,
            "player_2_last_shot_speed": 0,
            "player_2_total_player_speed": 0,
            "player_2_last_player_speed": 0,
        }
    ]

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1],
        )
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            cfg.COURT.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # player who the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            ),
        )

        # opponent player speed
        opponent_player_id = p1[0] if player_shot_ball == p2[0] else p2[0]
        if player_shot_ball == p2[0]:
            player_shot_ball = 2

        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id],
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            cfg.COURT.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court(),
        )

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        if opponent_player_id == p2[0]:
            opponent_player_id = 2
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball_shot

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on="frame_num", how="left")
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"] / player_stats_data_df["player_1_number_of_shots"]
    )
    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"] / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"] / player_stats_data_df["player_2_number_of_shots"]
    )
    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"] / player_stats_data_df["player_1_number_of_shots"]
    )

    # ANNOTATE THE VIDEO FRAMES
    ## Draw Player Bounding Boxes
    annotated_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    ## Draw Tennis Ball Bounding Box
    annotated_frames = ball_tracker.draw_bboxes(annotated_frames, ball_detections)

    ## Draw Court Keypoints (KPS detected on the first frame)
    annotated_frames = court_kps_detector.draw_keypoints_on_video(annotated_frames, court_kps)

    ## Draw MiniCourt (Translucent Canvas Background + Court)
    annotated_frames = mini_court.draw_mini_court(annotated_frames)

    ## Draw points (players+tennis ball) on the Minicourt
    annotated_frames = mini_court.draw_points_on_mini_court(
        annotated_frames,
        player_mini_court_detections,
    )
    annotated_frames = mini_court.draw_points_on_mini_court(
        annotated_frames,
        ball_mini_court_detections,
        color=(0, 255, 255),
    )

    # Draw Player Stats
    annotated_frames = draw_player_stats(annotated_frames, player_stats_data_df)

    # Draw frame number of the top-left corner of the video
    for i, frame in enumerate(annotated_frames):
        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    # Save the annotated video
    save_video(
        output_video_frames=annotated_frames,
        output_video_path="output_videos/V6.avi",
    )


if __name__ == "__main__":
    main()
