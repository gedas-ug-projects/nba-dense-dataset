Q: do we care about anything other than the clip annotations?
    A: why would we, also ask Jack & Rana
 
### TODO
0. Prepare updates for tomorrow's meeting @10:30
1. process and replace all files in `filtered-annotations` dirs
    - speed this process up?
2. update annoations to contain corrected `video_fp` values
3. extract time-remaining values from clips
4. map time-remaining values to statvu moments
5. update clip annotations

### All NBA Densely Annotated Dataset Features
1. [4d pose estimates (3d pose + body shape)](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file)
2. [3d pose estimates](https://github.com/ViTAE-Transformer/ViTPose)
3. 2d positions w/ ids
4. tracklets + 3d poses w/ ids (i.e. player re-id)
5. video transcripts / play-by-play commentary

**If we only care about clips, then there is no need to perform inference on an entire video.**