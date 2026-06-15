from typing import Union
from fractions import Fraction
import datetime
import av


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert non-skip frame timecode into seconds since midnight
    """
    # calculate whole frame rate
    # 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string
    h, m, s, f = [int(x) for x in timecode.split(':')]

    # calculate frames assuming whole frame rate (i.e. non-drop frame)
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata['timecode']
    try:
        creation_time = stream.metadata['creation_time']
    except KeyError:
        # 여기에 경고 메시지를 띄웁니다!
        print("⚠️ [WARNING] 'creation_time' metadata not found. Using default: 2026-01-15T12:00:00.000000Z")
        
        # 임의의 날짜로 강제 설정
        creation_time = "2026-01-15T12:00:00.000000Z"
    
    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = datetime.timedelta(seconds=seconds_since_midnight)
    
    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)
