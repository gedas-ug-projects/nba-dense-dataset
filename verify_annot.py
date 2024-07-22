import json
import cv2
import os
import random
from typing import List
from annot_types import VideoAnnotation, Bbox, Tracklet

def draw_annotations_on_frame(frame, bboxes: List[Bbox], tracklet: Tracklet):
    # Draw bounding boxes
    for bbox in bboxes:
        top_left = (int(bbox.x), int(bbox.y))
        bottom_right = (int(bbox.x + bbox.width), int(bbox.y + bbox.height))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    
    # Draw tracklet as a dot
    # if tracklet:
    #     for pos in tracklet.moment.player_positions:
    #         center = (int(pos.x_position), int(pos.y_position))
    #         cv2.circle(frame, center, 5, (0, 0, 255), -1)
    
    return frame

def process_video_with_annotations(annotation_path, output_folder):
    with open(annotation_path, 'r') as f:
        video_annotation = VideoAnnotation.parse_raw(f.read())
    period = video_annotation.video_path.split('_')[1]
    #video_path = f'/mnt/mir/fan23j/data/nba-plus-statvu-dataset/filtered-clips/17601/period1/17601_period1_1+_77129201_0.mp4'
    video_path = f'/mnt/mir/fan23j/data/nba-plus-statvu-dataset/clips/17601/period1/17601_period1_1+_77129201.mp4'
    #video_path = f'filtered-clips/17601/{period}/{video_annotation.video_path}'
    #video_path = 'game-replays/17993_11-22-2015_1_Cleveland Cavaliers_79_Atlanta Hawks_period3.mp4'
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_path)
    # Randomly sample 2 unique frame indices
    sampled_frame_indices = sorted(random.sample(range(total_frames), 2))
    print(f"Processing {annotation_path}")
    print(f"Sampled frame indices: {sampled_frame_indices}")

    frame_idx = 0
    sampled_frame_counter = 0

    while cap.isOpened() and sampled_frame_counter < 2:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sampled_frame_indices:
            if frame_idx < len(video_annotation.frames):
                frame_annotation = video_annotation.frames[frame_idx]
                frame = draw_annotations_on_frame(frame, frame_annotation.bbox, frame_annotation.tracklet)
            
            # Save frame as image
            frame_filename = os.path.join(output_folder, f"{os.path.basename(annotation_path)[:-5]}_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            sampled_frame_counter += 1

        frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()

def main():
    annotations_folder = 'clip-annotations/17601/period1'  # Replace with the actual annotations folder path
    #annotations_folder = 'annotations'
    output_folder = 'output_frames_orig'  # Replace with desired output folder

    os.makedirs(output_folder, exist_ok=True)

    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
    
    for annotation_file in annotation_files:
        annotation_file = '17601_period1_1+_77129201_annotation.json'
        annotation_path = os.path.join(annotations_folder, annotation_file)
        process_video_with_annotations(annotation_path, output_folder)

if __name__ == '__main__':
    main()