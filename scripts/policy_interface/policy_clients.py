from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable
from collections import deque
import contextlib
import functools
import signal
import numpy as np
import msgpack
import msgpack_numpy as mnp
import requests
import zmq

from openpi_client import image_tools
from openpi_client import websocket_client_policy
import json_numpy

json_numpy.patch()


class PolicyClient(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def infer(self, observation: dict, instruction: str, selected_camera: str = "left") -> np.ndarray:
        """Run inference and return a predicted action chunk (array of actions)."""
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def action_space(self) -> str:
        pass

    @abstractmethod
    def gripper_space(self) -> str:
        pass

    @contextlib.contextmanager
    def prevent_keyboard_interrupt(self):
        """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
        interrupted = False
        original_handler = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGINT, handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if interrupted:
                raise KeyboardInterrupt


class OpenPiClient(PolicyClient):
    """Client for pi0 / pi0.5 policies served via the openpi WebSocket server."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        if self.client is None:
            self.client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)

    def disconnect(self):
        self.client = None

    def infer(self, observation: dict, instruction: str, selected_camera: str = "left") -> np.ndarray:
        if self.client is None:
            raise RuntimeError("Client is not connected. Call connect() first.")

        request_data = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                observation[f"{selected_camera}_image"], 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                observation["wrist_image"], 224, 224
            ),
            "observation/joint_position": observation["joint_position"],
            "observation/gripper_position": observation["gripper_position"],
            "prompt": instruction,
        }

        with self.prevent_keyboard_interrupt():
            try:
                return self.client.infer(request_data)["actions"]
            except Exception:
                print("Disconnected — attempting to reconnect.")
                self.connect()
                return self.client.infer(request_data)["actions"]

    def action_space(self) -> str:
        return "joint_velocity"

    def gripper_space(self) -> str:
        return "position"


class MolmoActClient(PolicyClient):
    """Client for MolmoAct policies served via an HTTP REST server."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"

    def connect(self):
        try:
            response = requests.get(f"{self._base_url}/health", timeout=5)
            response.raise_for_status()
            print(f"MolmoAct server reachable at {self._base_url}")
        except requests.exceptions.ConnectionError:
            print(
                f"Warning: could not reach MolmoAct server at {self._base_url}. "
                "Continuing anyway — connection will be retried on first inference."
            )
        except requests.exceptions.HTTPError:
            pass  # Server is up but has no /health endpoint — that's fine.

    def disconnect(self):
        pass  # No persistent connection to close.

    def infer(self, observation: dict, instruction: str, selected_camera: str = "left") -> np.ndarray:
        payload = json_numpy.dumps({
            "external_cam": observation[f"{selected_camera}_image"],
            "wrist_cam": observation["wrist_image"],
            "instruction": instruction,
            "state": np.concatenate([observation["joint_position"], observation["gripper_position"]]),
        })

        with self.prevent_keyboard_interrupt():
            response = requests.post(
                f"{self._base_url}/act",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = json_numpy.loads(response.text)

        return result["actions"]

    def action_space(self) -> str:
        return "joint_position"

    def gripper_space(self) -> str:
        return "position"


class GRootClient(PolicyClient):
    """Client for GR00T policies served via a ZMQ server."""

    _EEF_ROTATION_CORRECT = np.array(
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64
    )
    _IMAGE_RESOLUTION = (180, 320)  # (H, W)

    def __init__(self, host: str, port: int, timeout_ms: int = 15000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket = None
        self._modality_keys = None
        self._video_delta = None

    def connect(self):
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

        self._send({"endpoint": "ping"})

        modality_config = self._send({"endpoint": "get_modality_config"})
        self._modality_keys = {
            modality: modality_config[modality]["modality_keys"]
            for modality in ["video", "state", "action", "language"]
        }
        self._video_delta = modality_config["video"]["delta_indices"]
        video_history_len = max(-min(self._video_delta), 0) + 1 if self._video_delta else 1
        self._frame_buffer = deque(maxlen=video_history_len)

    def disconnect(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._modality_keys = None
        self._video_delta = None
        self._frame_buffer = None

    def infer(self, observation: dict, instruction: str, selected_camera: str = "left") -> np.ndarray:
        if self._socket is None or self._modality_keys is None or self._frame_buffer is None:
            raise RuntimeError("Client is not connected. Call connect() first.")

        H, W = self._IMAGE_RESOLUTION
        ext_image = image_tools.resize_with_pad(observation[f"{selected_camera}_image"], H, W)
        wrist_image = image_tools.resize_with_pad(observation["wrist_image"], H, W)
        self._frame_buffer.append({"ext": ext_image, "wrist": wrist_image})

        obs = self._format_observation(observation, instruction)

        with self.prevent_keyboard_interrupt():
            action_chunk, _ = self._send(
                {"endpoint": "get_action", "data": {"observation": obs, "options": None}}
            )

        return np.concatenate([
            action_chunk["joint_position"][0],
            action_chunk["gripper_position"][0],
        ], axis=-1)

    def action_space(self) -> str:
        return "joint_position"

    def gripper_space(self) -> str:
        return "position"

    def _format_observation(self, observation: dict, instruction: str) -> dict:
        """Convert the standard observation dict into the nested format the GR00T server expects."""
        obs = {"video": {}, "state": {}, "language": {}}
        if self._frame_buffer is None:
            raise RuntimeError("Frame buffer is not initialized.")
        if self._modality_keys is None:
            raise RuntimeError("Modality keys are not initialized.")

        video_T = len(self._video_delta) if self._video_delta is not None else 1
        for key in self._modality_keys["video"]:
            if video_T == 1:
                
                frame = self._frame_buffer[-1]
                
                img = frame["wrist"] if "wrist" in key else frame["ext"]
                obs["video"][key] = img[None, None]
            else:
                hist = self._frame_buffer[0]
                cur = self._frame_buffer[-1]
                if hist is None or cur is None:
                    raise RuntimeError("Frame history contains None; cannot format video observation.")
                if "wrist" in key:
                    obs["video"][key] = np.stack([hist["wrist"], cur["wrist"]])[None]
                else:
                    obs["video"][key] = np.stack([hist["ext"], cur["ext"]])[None]

        state_source = {
            "eef_9d": self._compute_eef_9d(observation["cartesian_position"]),
            "gripper_position": observation["gripper_position"],
            "joint_position": observation["joint_position"],
        }
        for key in self._modality_keys["state"]:
            val = state_source[key][None, None, ...].astype(np.float32)
            obs["state"][key] = val

        lang_key = self._modality_keys["language"][0]
        obs["language"][lang_key] = [[instruction]]

        return obs

    @staticmethod
    def _compute_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
        """Convert XYZ + extrinsic XYZ Euler to XYZ + rot6d, corrected for OXE DROID convention."""
        from scipy.spatial.transform import Rotation
        c = np.asarray(cartesian_position, dtype=np.float64).reshape(6)
        rot_mat = Rotation.from_euler("XYZ", c[3:6]).as_matrix() @ GRootClient._EEF_ROTATION_CORRECT
        rot6d = rot_mat[:2, :].reshape(6)
        return np.concatenate([c[:3], rot6d]).astype(np.float32)

    def _send(self, request: dict):
        if self._socket is None:
            raise RuntimeError("Client is not connected. Call connect() first.")
        self._socket.send(GRootClient._MsgSerializer.to_bytes(request))
        response = GRootClient._MsgSerializer.from_bytes(self._socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    class _MsgSerializer:
        """msgpack + numpy serialization for ZMQ transport."""

        @staticmethod
        def to_bytes(data: Any) -> bytes:
            default = functools.partial(
                GRootClient._MsgSerializer._safe_encode,
                chain=GRootClient._MsgSerializer._encode_custom,
            )
            return msgpack.packb(data, default=default) or b""

        @staticmethod
        def from_bytes(data: bytes) -> Any:
            object_hook = functools.partial(
                GRootClient._MsgSerializer._safe_decode,
                chain=GRootClient._MsgSerializer._decode_custom,
            )
            return msgpack.unpackb(data, object_hook=object_hook, raw=False)

        @staticmethod
        def _safe_encode(obj, chain=None):
            if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
                raise TypeError(
                    f"Refusing to encode object-dtype ndarray (shape={obj.shape}); "
                    "msgpack_numpy would invoke pickle."
                )
            return mnp.encode(obj, chain=chain)

        @staticmethod
        def _safe_decode(obj, chain=None):
            if isinstance(obj, dict):
                nd_val = obj.get(b"nd", obj.get("nd"))
                kind_val = obj.get(b"kind", obj.get("kind"))
                if nd_val and kind_val in (b"O", "O"):
                    raise ValueError("Refusing to decode object-dtype ndarray payload.")
            return mnp.decode(obj, chain=chain)

        @staticmethod
        def _encode_custom(obj):
            if is_dataclass(obj) and not isinstance(obj, type):
                return {"__dataclass__": type(obj).__name__, "fields": asdict(obj)}
            if isinstance(obj, Enum):
                return {"__enum__": type(obj).__name__, "value": obj.value}
            raise TypeError(f"Cannot encode object of type {type(obj)}")

        @staticmethod
        def _decode_custom(obj):
            if not isinstance(obj, dict):
                return obj
            if "__dataclass__" in obj or b"__dataclass__" in obj:
                key = next((k for k in ("fields", b"fields") if k in obj), None)
                if key is None:
                    raise ValueError("Malformed dataclass payload: 'fields' missing.")
                return obj[key]
            return obj


# Registry: map policy names to their client classes.
POLICY_CLIENTS: "dict[str, Callable[[str, int], PolicyClient]]" = {
    "pi0": OpenPiClient,
    "pi05": OpenPiClient,
    "molmoact": MolmoActClient,
    "groot": GRootClient,
}
