from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError, validator, field_validator
from enum import Enum


class Keypoints(BaseModel):
    """
    Object representing all keypoints for a single bbx.
    133 total keypoints (COCO whole-body).
    Annotations given by: https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md
    """

    keypoints: List[List[float, float, float]]
    wholebody = {
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle",
            17: "left_big_toe",
            18: "left_small_toe",
            19: "left_heel",
            20: "right_big_toe",
            21: "right_small_toe",
            22: "right_heel",
            23: "face-0",
            24: "face-1",
            25: "face-2",
            26: "face-3",
            27: "face-4",
            28: "face-5",
            29: "face-6",
            30: "face-7",
            31: "face-8",
            32: "face-9",
            33: "face-10",
            34: "face-11",
            35: "face-12",
            36: "face-13",
            37: "face-14",
            38: "face-15",
            39: "face-16",
            40: "face-17",
            41: "face-18",
            42: "face-19",
            43: "face-20",
            44: "face-21",
            45: "face-22",
            46: "face-23",
            47: "face-24",
            48: "face-25",
            49: "face-26",
            50: "face-27",
            51: "face-28",
            52: "face-29",
            53: "face-30",
            54: "face-31",
            55: "face-32",
            56: "face-33",
            57: "face-34",
            58: "face-35",
            59: "face-36",
            60: "face-37",
            61: "face-38",
            62: "face-39",
            63: "face-40",
            64: "face-41",
            65: "face-42",
            66: "face-43",
            67: "face-44",
            68: "face-45",
            69: "face-46",
            70: "face-47",
            71: "face-48",
            72: "face-49",
            73: "face-50",
            74: "face-51",
            75: "face-52",
            76: "face-53",
            77: "face-54",
            78: "face-55",
            79: "face-56",
            80: "face-57",
            81: "face-58",
            82: "face-59",
            83: "face-60",
            84: "face-61",
            85: "face-62",
            86: "face-63",
            87: "face-64",
            88: "face-65",
            89: "face-66",
            90: "face-67",
            91: "left_hand_root",
            92: "left_thumb1",
            93: "left_thumb2",
            94: "left_thumb3",
            95: "left_thumb4",
            96: "left_forefinger1",
            97: "left_forefinger2",
            98: "left_forefinger3",
            99: "left_forefinger4",
            100: "left_middle_finger1",
            101: "left_middle_finger2",
            102: "left_middle_finger3",
            103: "left_middle_finger4",
            104: "left_ring_finger1",
            105: "left_ring_finger2",
            106: "left_ring_finger3",
            107: "left_ring_finger4",
            108: "left_pinky_finger1",
            109: "left_pinky_finger2",
            110: "left_pinky_finger3",
            111: "left_pinky_finger4",
            112: "right_hand_root",
            113: "right_thumb1",
            114: "right_thumb2",
            115: "right_thumb3",
            116: "right_thumb4",
            117: "right_forefinger1",
            118: "right_forefinger2",
            119: "right_forefinger3",
            120: "right_forefinger4",
            121: "right_middle_finger1",
            122: "right_middle_finger2",
            123: "right_middle_finger3",
            124: "right_middle_finger4",
            125: "right_ring_finger1",
            126: "right_ring_finger2",
            127: "right_ring_finger3",
            128: "right_ring_finger4",
            129: "right_pinky_finger1",
            130: "right_pinky_finger2",
            131: "right_pinky_finger3",
            132: "right_pinky_finger4",
        },
        "skeleton": [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [15, 17],
            [15, 18],
            [15, 19],
            [16, 20],
            [16, 21],
            [16, 22],
            [91, 92],
            [92, 93],
            [93, 94],
            [94, 95],
            [91, 96],
            [96, 97],
            [97, 98],
            [98, 99],
            [91, 100],
            [100, 101],
            [101, 102],
            [102, 103],
            [91, 104],
            [104, 105],
            [105, 106],
            [106, 107],
            [91, 108],
            [108, 109],
            [109, 110],
            [110, 111],
            [112, 113],
            [113, 114],
            [114, 115],
            [115, 116],
            [112, 117],
            [117, 118],
            [118, 119],
            [119, 120],
            [112, 121],
            [121, 122],
            [122, 123],
            [123, 124],
            [112, 125],
            [125, 126],
            [126, 127],
            [127, 128],
            [112, 129],
            [129, 130],
            [130, 131],
            [131, 132],
        ],
    }


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
    keypoints: Optional[Keypoints]


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

    @field_validator("*", pre=True)
    def empty_str_to_none(cls, v):
        if v == "":
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
