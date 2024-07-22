import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ray
import logging

columns = [
    'id', 'action_id', 'action_name', 'player_id', 'player_name',
    'team_id', 'team_name', 'opponent_id', 'opponent_name',
    'opponent_team_id', 'opponent_team_name', 'teammate_id',
    'teammate_name', 'half', 'second', 'pos_x', 'pos_y',
    'possession_id', 'possession_name', 'possession_team_id',
    'possession_team_name', 'possession_number', 'possession_start_clear',
    'possession_end_clear', 'playtype', 'hand', 'shot_type',
    'drive', 'dribble_move', 'contesting', 'ts'
]

ray.init(configure_logging=True, logging_level=logging.ERROR)

@ray.remote
def process_video(video, video_path, log_path, save_path, ignore):
    if video == '.DS_Store':
        return

    game_id = int(video.split('_')[0])
    period_id = video.split('_')[6].split('.')[0]
    curr_video = os.path.join(video_path, video)
    
    log_file = next((log for log in os.listdir(log_path) if str(game_id) in log), None)
    if log_file is None:
        return False

    log_df = pd.read_csv(os.path.join(log_path, log_file), skiprows=1, delimiter=';', header=None, names=columns)

    extend_time = 3.5

    period = int(video[-5])
    for index, row in log_df.iterrows():
        if int(row['half']) != period:
            continue

        if row['action_name'] not in ignore and not pd.isna(row['second']) and not pd.isna(row['player_name']):
            if row['action_name'] == 'Assisting':
                if pd.isna(row['teammate_id']):
                    continue
                player_id = int(row['teammate_id'])
                player_name = row['teammate_name']
                duration = 10
            else:
                if '1' in row['action_name']:
                    extend_time = 4.5
                elif row['action_name'] == 'Turnover':
                    extend_time = 2.5
                player_id = int(row['player_id'])
                player_name = row['player_name']
                duration = 10
            
            start_time = float(row['second']) + extend_time

            output_folder = os.path.join(save_path, str(game_id), str(period_id))
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, f"{game_id}_{period_id}_{row['action_name']}_{row['id']}.mp4")

            if not os.path.exists(output_path):
                os.system(
                    f'ffmpeg -ss {start_time - duration} -t {duration} -hide_banner -loglevel error -n -i "{curr_video}" -vcodec copy -acodec copy "{output_path}"')

            if not os.path.exists(output_path):
                with open(f'{game_id}_failed_videos.txt', 'a+') as f:
                    f.write(f'{curr_video}\n')
                return False

    return True


if __name__ == "__main__":
    video_path = './game-replays'
    log_path = './hudl-game-logs'
    save_path = f'./clips'
    os.makedirs(save_path, exist_ok=True)

    ignore = ['Start of the offensive possession', 'Shooting guard', 'Guard', 'Center', 'Power forward', 'Forward',
              'Timeout', 'Halftime', '2nd quarter', 'Starting lineup', '3rd quarter', '1st quarter', '4th quarter',
              'Match end', 'Game stop', 'Ball in play']
    
    ignore.append('Error leading to goal')
    ignore.append('Accurate pass')

    videos = [video for video in os.listdir(video_path) if video != '.DS_Store']
    
    pbar = tqdm(total=len(videos))
    video_path = ray.put(video_path)
    log_path = ray.put(log_path)
    ignore = ray.put(ignore)
    save_path = ray.put(save_path)

    futures = [process_video.remote(video, video_path, log_path, save_path, ignore) for video in videos]

    num_failed = 0
    while len(futures):
        dones, futures = ray.wait(futures)
        for done in dones:
            if ray.get(done) == False:
                num_failed += 1
            if num_failed % 100 == 0 and num_failed > 0:
                print(f'# failed video: {num_failed}')
        pbar.update(len(dones))

    print(f'Total # failed video: {num_failed}')

    ray.shutdown()