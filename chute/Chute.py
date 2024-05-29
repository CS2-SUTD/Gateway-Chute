import time
import socket
import configparser
import json
import os
import ftplib
import pickle
from datetime import datetime, timezone, timedelta
from typing import List

import cv2
import paho.mqtt.client as mqtt
from loguru import logger
import numpy as np

from chute.Detector import Detector
from chute.Camera import Camera
from chute.utils import *


class Chute:
    """
    Chute object that process RTSP stream and publish results to backend

    Note: Configuration dictates the config for the detector, mqtt broker,
    ftp server and socket client to receive live stream frames.

    Attributes:
        config_path (str): path to ini file with configurations of Chute
    """

    def __init__(self, config_path: str = "chute/config.ini"):
        self.config_path: str = config_path
        det_cfg: DetCfg = self._get_config("detector")
        self.socket_cfg: SocketCfg = self._get_config("socket")
        self.ftp_cfg: FtpCfg = self._get_config("ftp")
        self.mqtt_cfg: MqttCfg = self._get_config("mqtt")
        self.class_names: List[str] = det_cfg["class_names"].split(",")
        self.detector = Detector(**det_cfg)
        self.open: bool = False
        self.socket = self._get_socket()

    def start(self, stream_url: str):
        """
        Start Chute processing of a RTSP stream

        Take in a RTSP stream, and continually within each cycle, catches
        the latest frame and runs the model on each frame. Pass the anotated
        frame to the backend via sockets for the live stream. Upon detecting
        open chute, start a timer of 15 seconds, if timer is up and chute is
        still open, will upload message via mqtt and the 15s video evidence via
        FTP. It will stop recording and upload anything until the chute has
        closed again.

        Args:
            stream_url (str): RTSP url
        """

        self.recording = False
        self.recorded = False
        self.frame_buffer = []
        self.is_stopping = False

        self.cam = Camera(stream_url)

        while True:
            frame = self.cam.read()
            bbox_xyxy, scores, class_ids = self.detector.detect(frame)
            try:
                self.open = class_ids[0]
            except:
                pass

            frame = draw_boxes(
                frame, bbox_xyxy, class_ids, class_names=self.class_names
            )

            # cv2.imshow("Live Video", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            self._send_frame(frame)
            # logger.info("Unable to send frame to server")

            if not self.recording and not self.recorded and self.open:
                self.box_info = bb_info(bbox_xyxy, 0)
                logger.info(f"Chute opened at {get_string_time()}")
                self.recording = True

            if self.recording:
                self.frame_buffer.append(frame)
                if not self.recorded:
                    self._record_irresponsible()

            if self.recorded and not self.open:
                logger.info(
                    f"Chute that has been opened for a long time has finally closed at {time.time() - self.time_to_stop + 40}"
                )
                self.recorded = False

    def _record_irresponsible(self):
        """
        Determines whether chute is being left open for a long time
        and executes the appropriate actions
        """

        if self.open:
            if not self.is_stopping:
                self.time_to_stop = time.time() + 10
                self.is_stopping = True
            else:
                if time.time() > self.time_to_stop:
                    self._upload_evidence()
                    self.recorded = True
                    self.recording = False
                    self.is_stopping = False
                    self.frame_buffer = []
        else:
            logger.info(f"Chute has closed at {get_string_time()}")
            self.is_stopping = False
            self.frame_buffer = []
            self.recording = False

    def _send_frame(self, frame: np.ndarray):
        """
        Send the frame to remote host

        Args:
            frame (np.ndarray): frame to be sent
        """
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 8])
        x_as_bytes = pickle.dumps(buffer)
        self.socket.sendto(
            (x_as_bytes), (self.socket_cfg["ip"], int(self.socket_cfg["port"]))
        )

    def _upload_evidence(self):
        """
        Upload evidence
        - send video evidence
        - send message of detected open chute for prolonged period
        """

        # now = datetime.now()
        current_time = datetime.now(tz=timezone(timedelta(hours=8)))
        time_iso = current_time.isoformat()  # ISO 8601 string
        timestamp = current_time.strftime("%Y-%m-%d_%H%M%S")
        # timestamp = f'{str(now.date())}_{str(now.hour)}{str(now.minute)}{str(now.second)}'
        path = f"{timestamp}.mp4"
        logger.info(f"Recording the evidence video {path}")
        output = cv2.VideoWriter(
            path, self.cam.fourcc, 12, (self.cam.width, self.cam.height)
        )
        for frame in self.frame_buffer:
            output.write(frame)
            k = cv2.waitKey(24)
            if k == ord("q"):
                break
        output.release()
        self._send_video(path)
        self._send_message(time_iso, path, self.box_info)

    def _send_message(self, time_iso: str, path: str, Box_info: BBxywh):
        """
        Send message regarding chute not closed for prolonged period to mqtt broker
        """

        message_dict = {
            "Id": path,
            "VideoSource": {
                "Name": "camera1",
                "Type": "Network",
                "URL": None,
            },
            "Date": time_iso,
            "Image": "Lab camera",
            "Video": "Lab camera",
            "Category": "VA",
            "Metadata": {"State": "Open", "Box": Box_info},
        }

        message_json = json.dumps(message_dict)
        client = mqtt.Client()
        try:
            client.connect(self.mqtt_cfg["ip"], int(self.mqtt_cfg["port"]), 60)
            client.loop_start()
            client.publish(
                self.mqtt_cfg["topic"], message_json, qos=0
            )  # qos could be alter to 1 in the future
            client.loop_stop()

        except Exception as e:
            logger.error(f'Error sending message: "{message_json}", Error: {str(e)}')
        finally:
            client.disconnect()
            logger.info(f'Sent object details: "{message_json}"')

    def _send_video(self, video_path: str):
        """
        Send video evidence to backend via ftp
        """

        filename = os.path.basename(video_path)
        cfg: FtpCfg = self._get_config("ftp")

        # Connect to the FTP server
        ftp = ftplib.FTP(cfg["host"], cfg["user"], cfg["passwd"])
        ftp.cwd(cfg["dir"])

        # Upload the video to Backend
        try:
            with open(video_path, "rb") as file:
                retcode = ftp.storbinary(
                    f"STOR {filename}", file, blocksize=1024 * 1024
                )

            if retcode.startswith("226"):
                logger.success(f"Video upload successful: {filename}")
                os.remove(video_path)
            else:
                logger.error(f"Video upload failed: {filename}")

        except:
            logger.error(f"Video upload error: {filename}")

        finally:
            ftp.quit()

    def _get_config(self, section: str):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        return config[section]

    def _get_socket(self):
        mysocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mysocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
        return mysocket


if __name__ == "__main__":
    chute = Chute()
    chute.start(0)
