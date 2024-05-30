from typing import TypedDict, Tuple


class DetCfg(TypedDict):
    weights: str
    input_shape: str
    class_names: str
    cpu_cores: str


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


class GeneralCfg(TypedDict):
    enable_server: str
    chute_timeout: str


BBxywh = Tuple[float, float, float, float]
