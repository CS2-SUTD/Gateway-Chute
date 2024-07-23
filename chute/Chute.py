import time
import socket
import configparser
import json
import os
import ftplib
import pickle
import threading
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

    def __init__(self, config_path: str = "config.ini"):
        self.config_path: str = config_path
        det_cfg: DetCfg = self._get_config("detector")
        general_cfg: GeneralCfg = self._get_config("general")
        self.socket_cfg: SocketCfg = self._get_config("socket")
        self.ftp_cfg: FtpCfg = self._get_config("ftp")
        self.mqtt_cfg: MqttCfg = self._get_config("mqtt")
        self.CLASS_NAMES: List[str] = det_cfg["class_names"].split(",")
        self.detector = Detector(**det_cfg)
        self.cam_id: str = general_cfg["cam_id"]
        self.server_enabled: bool = bool(int(general_cfg["enable_server"]))
        self.chute_timeout: int = int(general_cfg["chute_timeout"])
        self.socket = self._get_socket()

    def start(self, source: str):
        """
        Start Chute processing of a RTSP stream

        Note: Source format supported are streams and capturing devices. Local
        files can be used but algorithm to decide prolonged opening of chute
        may not function properly

        Take in a RTSP stream, and continually within each cycle, catches
        the latest frame and runs the model on each frame. Pass the anotated
        frame to the backend via sockets for the live stream. Upon detecting
        open chute, start a timer of 15 seconds, if timer is up and chute is
        still open, will upload message via mqtt and the 15s video evidence via
        FTP. It will stop recording and upload anything until the chute has
        closed again.

        Args:
            source (str | int): RTSP url, capturing device id, *.mp4, *.avi
        """
        self.recorded = False
        self.frame_buffer = []
        self.endtime_created = False
        self.cam = Camera(source)

        while True:
            frame = self.cam.read()
            bbox_xyxy, scores, class_ids = self.detector.detect(frame)
            try:
                if class_ids[0] == False and scores[0] < 0.85:
                    class_ids[0] = self.open
                self.open = int(class_ids[0])
                self.bbox_xyxy = bbox_xyxy
            except:
                pass

            frame = draw_boxes(
                frame, bbox_xyxy, class_ids, class_names=self.CLASS_NAMES
            )

            if self.server_enabled:
                self._send_frame(frame)
            else:
                cv2.imshow("Live Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if self.open:
                # Chute has just been opened from being closed. Create an endtime.
                if self.endtime_created == False and self.recorded == False: 
                    self.time_to_stop = time.time() + self.chute_timeout + 2
                    self.frame_buffer.append(frame)
                    self.endtime_created = True
                    logger.info(f"Chute has just been opened at {get_logging_time()}")

                # Chute has been left open for a very long time. 
                # Short video of the first (chute_timeout) seconds of being open has already been uploaded, so don't create a new endtime, as we don't want to create a new recording. 
                elif self.endtime_created == False and self.recorded == True:
                    continue

                else: 
                    # If chute was open and the duration of time it has been open has just exceeded chute_timeout, upload video evidence.  
                    if time.time() > self.time_to_stop and self.recorded == False:
                        self.frame_buffer.append(frame)
                        box_info = bb_info(self.bbox_xyxy, 0)
                        
                        # Start another thread for uploading video, if server is enabled.
                        if self.server_enabled:
                            logger.info(self.open)
                            t1 = threading.Thread(
                                target=self._upload_evidence,
                                args=(self.frame_buffer, box_info, self.open),
                            )
                            t1.start()
                            
                        # Continue in same thread for uploading video, if server is not enabled.    
                        else:
                            self._upload_evidence(
                                self.frame_buffer, box_info, self.open
                            )
                
                        self.recorded = True
                        self.endtime_created = False
                        self.frame_buffer = []

                    # If chute was open but the duration of time it has been open is still within chute_timeout, continue adding frames to buffer (in preparation for uploading video).
                    else:
                        self.frame_buffer.append(frame)
                
            else:
                # If chute that has been left open for a very long time (more than chute_timeout duration) has just been closed, reset all variables to original values.
                if self.endtime_created == False and self.recorded == True:
                    self.recorded = False
                    self.endtime_created = False
                    self.frame_buffer = []
                    logger.info(f"Chute that has been opened for a long time has finally closed at {get_logging_time()}")
                    logger.info(f"Duration for which chute was left open is: {round(time.time() - self.time_to_stop + self.chute_timeout, 1)} seconds")
                
                # If chute is "still closed", that is, the previous frame was still that of a closed chute, then don't do anything.  
                elif self.endtime_created == False and self.recorded == False:
                    continue
                
                else:
                    # If chute has just been closed (within chute_timeout duration of being opened), then discard current frame buffer. 
                    if time.time() < self.time_to_stop:
                        self.frame_buffer = []
                        self.endtime_created = False
                        logger.info(f"Chute has closed at {get_logging_time()}")

  
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

    def _upload_evidence(self, frame_buffer, box_info: BBxywh, state: int):
        """
        Upload evidence
        - send video evidence
        - send message of detected open chute for prolonged period
        """

        logger.info(f"Prolonged opening of chute detected")
        os.makedirs("data/evidence", exist_ok=True)
        evidence_path = f"data/evidence/{get_string_time()}.mp4"
        output = cv2.VideoWriter(
            evidence_path, self.cam.fourcc, 12, (self.cam.width, self.cam.height)
        )
        for frame in frame_buffer:
            output.write(frame)
            k = cv2.waitKey(24)
            if k == ord("q"):
                break
        output.release()
        logger.info(f"Evidence recorded and stored in {evidence_path}")
        if self.server_enabled:
            self._send_video(evidence_path)
            self._send_message(get_iso_time(), evidence_path, box_info, state)

    def _send_message(self, time_iso: str, path: str, Box_info: BBxywh, state: int):
        """
        Send message regarding chute not closed for prolonged period to mqtt broker
        """
        path = os.path.basename(path)
        message_dict = {
            "Id": path,
            "VideoSource": {
                "Name": f"camera {self.cam_id}",
                "Type": "Network",
                "URL": None,
            },
            "Date": time_iso,
            "Image": "Lab camera",
            "Video": "Lab camera",
            "Category": "VA",
            "Metadata": {"State": self.CLASS_NAMES[state], "Box": Box_info},
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
