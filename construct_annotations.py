import os
import json
import csv
from typing import List, Dict
from annot_types import Bbox, Tracklet, ActionAnnotation, FrameAnnotation, VideoAnnotation, ActionName


def load_2d_player_positions(file_path: str) -> Dict[int, Tracklet]:
    tracklets = {}
    with open(file_path, 'r') as f:
        data = json.load(f)
        for frame_number, tracklet_data in data.items():
            if tracklet_data:
                tracklet = Tracklet(frame_number=int(frame_number), **tracklet_data)
                tracklets[int(frame_number)] = tracklet
            else:
                tracklets[int(frame_number)] = None
    return tracklets

def load_hudl_game_logs(file_path: str) -> List[ActionAnnotation]:
    action_annotations = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                action_name = ActionName(row['action_name'])
                player_id = ActionName(row['player_id'])
            except ValueError:
                continue
            action_annotations.append(ActionAnnotation(
                annotation_id=int(row['id']),
                action_id=int(row['action_id']),
                action_name=ActionName(row['action_name']),
                player_id=int(row['player_id']),
                player_name=row['player_name'],
                team_id=int(row['team_id']),
                team_name=row['team_name'],
                half=int(row['half']),
                second=float(row['second']),
                pos_x=float(row['pos_x']) if row['pos_x'] else None,
                pos_y=float(row['pos_y']) if row['pos_y'] else None,
                opponent_id=int(row['opponent_id']) if row['opponent_id'] else None,
                opponent_name=row['opponent_name'] if row['opponent_name'] else None,
                opponent_team_id=int(row['opponent_team_id']) if row['opponent_team_id'] else None,
                opponent_team_name=row['opponent_team_name'] if row['opponent_team_name'] else None,
                teammate_id=int(row['teammate_id']) if row['teammate_id'] else None,
                teammate_name=row['teammate_name'] if row['teammate_name'] else None,
                possession_id=int(row['possession_id']) if row['possession_id'] else None,
                possession_name=row['possession_name'] if row['possession_name'] else None,
                possession_team_id=int(row['possession_team_id']) if row['possession_team_id'] else None,
                possession_team_name=row['possession_team_name'] if row['possession_team_name'] else None,
                possession_number=int(row['possession_number']) if row['possession_number'] else None,
                possession_start_clear=float(row['possession_start_clear']) if row['possession_start_clear'] else None,
                playtype=row['playtype'] if row['playtype'] else None,
                hand=row['hand'] if row['hand'] else None,
                shot_type=row['shot_type'] if row['shot_type'] else None,
                drive=row['drive'] if row['drive'] else None,
                dribble_move=row['dribble_move'] if row['dribble_move'] else None,
                contesting=row['contesting'] if row['contesting'] else None,
            ))
    return action_annotations


def load_player_bbox(file_path: str) -> dict:
    bboxes_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            frame_number = int(data[0])
            bbox = Bbox(
                frame_number=frame_number,
                player_id=int(data[1]),
                x=float(data[2]),
                y=float(data[3]),
                width=float(data[4]),
                height=float(data[5]),
                confidence=float(data[6])
            )
            if frame_number not in bboxes_dict:
                bboxes_dict[frame_number] = []
            bboxes_dict[frame_number].append(bbox)
    return bboxes_dict


def generate_video_annotation(video_id: int, video_path: str, quarter: str, data_dir: str) -> VideoAnnotation:
    quarter_map = {
        'period1': 'Q1',
        'period2': 'Q2',
        'period3': 'Q3',
        'period4': 'Q4'
    }

    period = quarter_map.get(quarter.lower(), quarter)

    player_positions_path = next(
        (os.path.join(data_dir, '2d-player-positions', file) for file in os.listdir(os.path.join(data_dir, '2d-player-positions')) if str(video_id) in file and period in file),
        None
    )
    
    player_bbox_paths = [
        os.path.join(data_dir, 'player-tracklets', file) for file in os.listdir(os.path.join(data_dir, 'player-tracklets')) if str(video_id) in file and quarter.lower() in file
    ]

    tracklets = {}
    if player_positions_path:
        tracklets = load_2d_player_positions(player_positions_path)
    else:
        print(f"Warning: 2D player positions file not found for video ID {video_id}, quarter {quarter}")

    bboxes_dict = {}
    if player_bbox_paths:
        for path in player_bbox_paths:
            bboxes_dict.update(load_player_bbox(path))
    else:
        print(f"Warning: Player bbox files not found for video ID {video_id}, quarter {quarter}")

    frames = []
    all_frame_numbers = sorted(set(list(tracklets.keys()) + list(bboxes_dict.keys())))

    for frame_number in all_frame_numbers:
        frames.append(FrameAnnotation(
            frame_id=frame_number,
            bbox=bboxes_dict.get(frame_number, []),
            tracklet=tracklets.get(frame_number)
        ))

    return VideoAnnotation(
        video_id=video_id,
        video_path=video_path,
        frames=frames,
        caption=f'Annotation for video {video_id}, {quarter}'
    )

def main():
    data_dir = '.'
    game_replays_dir = os.path.join(data_dir, 'game-replays')
    video_files = [f for f in os.listdir(game_replays_dir) if f.endswith('.mp4')]
    output_folder = 'annotations'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for video_file in video_files:
        video_id = int(video_file.split('_')[0])
        quarter = next(part for part in video_file.split('_') if part.startswith('period')).split('.')[0]
        video_path = os.path.join(game_replays_dir, video_file)
        
        # Check if the annotation file already exists
        output_file = f'{output_folder}/{video_id}_{quarter}_video_annotation.json'
        if os.path.exists(output_file):
            print(f'Annotation file for video ID {video_id}, quarter {quarter} already exists. Skipping.')
            continue

        try:
            video_annotation = generate_video_annotation(video_id, video_path, quarter, data_dir)
            with open(output_file, 'w') as f:
                json.dump(video_annotation.dict(), f, indent=4)
            print(f'Generated annotation for video ID {video_id}, quarter {quarter}')
        except FileNotFoundError as e:
            print(f'Error generating annotation for video ID {video_id}, quarter {quarter}: {e}')
        except Exception as e:
            print(f'Unexpected error for video ID {video_id}, quarter {quarter}: {e}')

if __name__ == '__main__':
    main()