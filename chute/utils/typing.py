from typing import TypedDict, Tuple


class DetCfg(TypedDict):
    weights: str
    input_shape: str
    class_names: str


class FtpCfg(TypedDict):
    host: str
    user: str
    passwd: str
    dir: str


class SocketCfg(TypedDict):
    ip: str
    port: str


class MqttCfg(TypedDict):
    ip: str
    port: str
    topic: str


BBxywh = Tuple[float, float, float, float]
