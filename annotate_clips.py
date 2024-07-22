import os
import json
import pandas as pd
import cv2

from glob import glob
from typing import List
from pydantic import BaseModel
from annot_types import VideoAnnotation, FrameAnnotation, ActionAnnotation, ActionName
from datetime import timedelta

# global fps value for all videos in our dataset
FPS = 30.0

# all column names for HUDL logs
columns = [
    "id",
    "action_id",
    "action_name",
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "opponent_id",
    "opponent_name",
    "opponent_team_id",
    "opponent_team_name",
    "teammate_id",
    "teammate_name",
    "half",
    "second",
    "pos_x",
    "pos_y",
    "possession_id",
    "possession_name",
    "possession_team_id",
    "possession_team_name",
    "possession_number",
    "possession_start_clear",
    "possession_end_clear",
    "playtype",
    "hand",
    "shot_type",
    "drive",
    "dribble_move",
    "contesting",
    "ts",
]


def load_video_annotation(file_path: str) -> VideoAnnotation:
    """
    Load an annotation file as found in the `annotations` dir.
    """
    
    assert os.path.isfile(file_path), f"{file_path} does not exist"
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise Exception(f"Failed to load annotation from {file_path}: {e}")
    
    # im such a noob, **data is meta
    return VideoAnnotation(**data)


def save_video_annotation(annotation: VideoAnnotation, file_path: str):
    """
    Save video annotation object to `file_path`.
    """
    
    assert os.path.exists(file_path), f"{file_path} does not exist"
    try:
        with open(file_path, "w") as f:
            json.dump(annotation.dict(), f, indent=4)
    except Exception as e:
        raise Exception(f"Failed to save annotation to {file_path}: {e}")


def split_video_annotation(
    video_annotation: VideoAnnotation, clip_info: dict, video_path: str
) -> VideoAnnotation:
    """
    Clip a video and return a new `VideoAnnotation` object.
    """
    
    # get start and end frames of a clip
    start_time = clip_info["start_time"]
    duration = clip_info["duration"]

    start_frame = int((start_time - duration) * FPS)
    end_frame = int(start_time * FPS)

    clip_frames = []
    for frame in video_annotation.frames:
        
        # what is this frame_id attribute?
        if start_frame <= frame.frame_id < end_frame:
            new_frame = frame.model_copy(deep=True)
            new_frame.frame_id -= start_frame

            # adjust bbox frame numbers
            if new_frame.bbox:
                for bbox in new_frame.bbox:
                    bbox.frame_number -= start_frame

            # adjust tracklet frame number if it exists
            if new_frame.tracklet:
                new_frame.tracklet.frame_number -= start_frame

            clip_frames.append(new_frame)

    # Save start and end frames as JPEGs
    # video = cv2.VideoCapture(os.path.join('game-replays',video_path))
    # video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # ret, frame = video.read()
    # if ret:
    #     cv2.imwrite(f"debug_{clip_info['action_id']}_start_frame.jpg", frame)

    # video.set(cv2.CAP_PROP_POS_FRAMES, end_frame - 1)
    # ret, frame = video.read()
    # if ret:
    #     cv2.imwrite(f"debug_{clip_info['action_id']}_end_frame.jpg", frame)

    # video.release()

    # create a new video annotation obj
    return VideoAnnotation(
        video_id=int(f"{video_annotation.video_id}{clip_info['action_id']}"),
        video_path=clip_info["output_path"],
        frames=clip_frames,
        caption=f"{clip_info['action_name']} by {clip_info['player_name']}",
    )


def process_annotations(
    video_path: str, log_path: str, annotation_path: str, output_path: str
):
    """
    
    """
    
    ignore = [
        "Start of the offensive possession",
        "Shooting guard",
        "Guard",
        "Center",
        "Power forward",
        "Forward",
        "Timeout",
        "Halftime",
        "2nd quarter",
        "Starting lineup",
        "3rd quarter",
        "1st quarter",
        "4th quarter",
        "Match end",
        "Game stop",
        "Ball in play",
        "Error leading to goal",
        "Accurate pass",
    ]

    
    video_file_names = [os.path.basename(f) for f in glob(f"{video_path}/*.mp4")]
    for video_file in video_file_names:

        game_id = int(video_file.split("_")[0])
        period_id = video_file.split("_")[6].split(".")[0]

        # Load the corresponding log file
        log_file = next(
            (log for log in os.listdir(log_path) if str(game_id) in log), None
        )
        if log_file is None:
            continue

        log_df = pd.read_csv(
            os.path.join(log_path, log_file),
            skiprows=1,
            delimiter=";",
            header=None,
            names=columns,
        )

        # load the corresponding annotation file
        annotation_file = f"{game_id}_{period_id}_video_annotation.json"
        annotation_path_full = os.path.join(annotation_path, annotation_file)
        assert os.path.exists(annotation_path_full), f"{annotation_path_full} does not exist"
        
        if not os.path.exists(annotation_path_full):
            continue

        video_annotation = load_video_annotation(annotation_path_full)
        extend_time = 3.5
        period = int(video_file[-5])
        for _, row in log_df.iterrows():
            if int(row["half"]) != period:
                continue

            if (
                row["action_name"] not in ignore
                and not pd.isna(row["second"])
                and not pd.isna(row["player_name"])
            ):

                duration = 10

                if row["action_name"] == "Assisting":
                    if pd.isna(row["teammate_id"]):
                        continue
                    player_id = int(row["teammate_id"])
                    player_name = row["teammate_name"]
                elif "1" in row["action_name"]:
                    extend_time = 4.5
                    player_id = int(row["player_id"])
                    player_name = row["player_name"]
                elif row["action_name"] == "Turnover":
                    extend_time = 2.5
                    player_id = int(row["player_id"])
                    player_name = row["player_name"]
                else:
                    player_id = int(row["player_id"])
                    player_name = row["player_name"]

                start_time = float(row["second"]) + extend_time

                clip_info = {
                    "start_time": start_time,
                    "duration": duration,
                    "action_id": row["id"],
                    "action_name": row["action_name"],
                    "player_name": player_name,
                    "output_path": f"{game_id}_{period_id}_{row['action_name']}_{row['id']}.mp4",
                }

                clip_annotation = split_video_annotation(
                    video_annotation, clip_info, video_file
                )

                pos_x = row["pos_x"] if not pd.isna(row["pos_x"]) else None
                pos_y = row["pos_y"] if not pd.isna(row["pos_y"]) else None

                # Helper function to convert to int or None
                def to_int_or_none(value):
                    return int(value) if pd.notna(value) and value != "" else None

                # Create ActionAnnotation
                action_annotation = ActionAnnotation(
                    id=to_int_or_none(row["id"]),
                    action_id=(
                        str(row["action_id"]) if pd.notna(row["action_id"]) else None
                    ),
                    action_name=row["action_name"],
                    player_id=str(player_id),
                    player_name=player_name,
                    team_id=to_int_or_none(row["team_id"]),
                    team_name=row["team_name"],
                    opponent_id=to_int_or_none(row["opponent_id"]),
                    opponent_name=row["opponent_name"],
                    opponent_team_id=to_int_or_none(row["opponent_team_id"]),
                    opponent_team_name=row["opponent_team_name"],
                    teammate_id=(
                        str(row["teammate_id"])
                        if pd.notna(row["teammate_id"])
                        else None
                    ),
                    teammate_name=row["teammate_name"],
                    half=to_int_or_none(row["half"]),
                    second=float(row["second"]) if pd.notna(row["second"]) else None,
                    pos_x=pos_x,
                    pos_y=pos_y,
                    possession_id=to_int_or_none(row["possession_id"]),
                    possession_name=row["possession_name"],
                    possession_team_id=to_int_or_none(row["possession_team_id"]),
                    possession_team_name=row["possession_team_name"],
                    possession_number=to_int_or_none(row["possession_number"]),
                    possession_start_clear=row["possession_start_clear"],
                    possession_end_clear=row["possession_end_clear"],
                    playtype=row["playtype"],
                    hand=row["hand"],
                    shot_type=row["shot_type"],
                    drive=row["drive"],
                    dribble_move=row["dribble_move"],
                    contesting=row["contesting"],
                    ts=row["ts"],
                )

                # Add ActionAnnotation to the clip annotation
                clip_annotation.action = action_annotation

                # Save the clip annotation
                output_folder = os.path.join(output_path, str(game_id), str(period_id))
                os.makedirs(output_folder, exist_ok=True)
                output_file = os.path.join(
                    output_folder,
                    f"{clip_info['output_path'].replace('.mp4', '_annotation.json')}",
                )
                save_video_annotation(clip_annotation, output_file)


if __name__ == "__main__":
    video_path = "./game-replays"
    log_path = "./hudl-game-logs"
    annotation_path = "./annotations"
    output_path = "./clip-annotations"

    process_annotations(video_path, log_path, annotation_path, output_path)
