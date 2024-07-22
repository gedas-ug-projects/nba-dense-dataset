from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError, validator, field_validator
from enum import Enum


class Bbox(BaseModel):
    """
    The bounding box of a player taken as a snapshot of a tracklet at a given frame.
    These bounding boxes are estimated by MixSort and do not contain meaningful 
    `player_id` values at the moment.
    """
    
    frame_number: int
    player_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float


class Position(BaseModel):
    """
    Position of a player or basketball in 2D space
    - x_position: [0-100]
    - y_position: [0-50]
    - z_position: [for ball only]
    Note that a ball obj will always have a `team_id` and `player_id` of -1
    """
    
    team_id: int
    player_id: int
    x_position: float
    y_position: float
    z_position: float


class Moment(BaseModel):
    """
    Object given by statvu data that contains the positions of players and the ball at 
    the `quarter` and `time_remaining_in_quarter` in a game.
    """

    quarter: int
    moment_id: int
    time_remaining_in_quarter: float
    time_remaining_on_shot_clock: Optional[float]
    player_positions: List[Position]


class Tracklet(BaseModel):
    """
    Mapping between frame number and Moment objects via OCR.
    """
    
    frame_number: int
    pred_quarter: str
    pred_time_remaining: float
    moment: Moment


class ActionName(str, Enum):
    MISSED_THREE_POINTER = "3-"
    ASSISTING = "Assisting"
    SCREEN = "Screen"
    REBOUND = "Rebound"
    TURNOVER = "Turnover"
    MADE_SINGLE_FREETHROW = "1+"
    MISSED_SINGLE_FREETHROW = "1-"
    AND_ONE = "2+1"
    MISSED_TWO_POINTER = "2-"
    MADE_TWO_POINTER = "2+"
    FOUL = "Foul"
    PICK_N_ROLL = "Pick'n'Roll"
    POST = "Post"
    STEAL = "Steal"
    TECHNICAL_FOUL = "Technical foul"
    MADE_THREE_POINTER = "3+"
    SECOND_FOUL = "2F"
    THIRD_FOUL = "3F"
    UNSPORTMANLIKE_FOUL = "Unsportmanlike foul"
    THREE_PLUS_ONE = "3+1"
    SECOND_CHANCE = "Second chance"
    MADE_TWO_FREETHROWS = "2FT+"
    MISSED_TWO_FREETHROWS = "2FT-"
    MADE_THREE_FREETHROWS = "3FT+"
    MISSED_THREE_FREETHROWS = "3FT-"
    DISQUALIFYING_FOUL = "Disqualifying foul"


class ActionAnnotation(BaseModel):
    """
    Object representing a row from a csv file in the `hudl-game-logs` dir.
    """
    
    id: Optional[int] = None
    action_id: Optional[str] = None
    action_name: Optional[str] = None
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    opponent_id: Optional[int] = None
    opponent_name: Optional[str] = None
    opponent_team_id: Optional[int] = None
    opponent_team_name: Optional[str] = None
    teammate_id: Optional[str] = None
    teammate_name: Optional[str] = None
    half: Optional[int] = None
    second: Optional[float] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None
    possession_id: Optional[int] = None
    possession_name: Optional[str] = None
    possession_team_id: Optional[int] = None
    possession_team_name: Optional[str] = None
    possession_number: Optional[int] = None
    possession_start_clear: Optional[float] = None
    possession_end_clear: Optional[str] = None
    playtype: Optional[str] = None
    hand: Optional[str] = None
    shot_type: Optional[str] = None
    drive: Optional[str] = None
    dribble_move: Optional[str] = None
    contesting: Optional[str] = None
    ts: Optional[str] = None

    @field_validator('*', pre=True)
    def empty_str_to_none(cls, v):
        if v == '':
            return None
        return v


class FrameAnnotation(BaseModel):
    """
    All annotations for a single frame in a video.
    """
    
    frame_id: int
    bbox: Optional[List[Bbox]] = []
    tracklet: Optional[Tracklet] = None


class VideoAnnotation(BaseModel):
    """
    All annotations for a single video.
    Optionaly include a video `caption` and `action`.
    """
    
    video_id: int
    video_path: str
    frames: List[FrameAnnotation]
    caption: Optional[str] = None
    action: Optional[ActionAnnotation] = None