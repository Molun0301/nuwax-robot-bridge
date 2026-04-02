from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any

from settings import DoubaoConfig


TTS_BIDIRECTIONAL_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

PROTOCOL_VERSION = 0x1
HEADER_WORDS = 0x1
HEADER_SIZE = HEADER_WORDS * 4
FLAG_WITH_EVENT = 0x4


class MessageType(IntEnum):
    FULL_CLIENT_REQUEST = 0x1
    AUDIO_ONLY_REQUEST = 0x2
    FULL_SERVER_RESPONSE = 0x9
    AUDIO_ONLY_RESPONSE = 0xB
    FRONTEND_SERVER_RESPONSE = 0xC
    ERROR_INFORMATION = 0xF


class Serialization(IntEnum):
    RAW = 0x0
    JSON = 0x1


class Compression(IntEnum):
    NONE = 0x0
    GZIP = 0x1


class Event(IntEnum):
    START_CONNECTION = 1
    FINISH_CONNECTION = 2

    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52

    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102

    SESSION_STARTED = 150
    SESSION_CANCELLED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153

    TASK_REQUEST = 200

    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    TTS_SUBTITLE = 354


CONNECT_REQUEST_EVENTS = {
    Event.START_CONNECTION,
    Event.FINISH_CONNECTION,
}

CONNECT_RESPONSE_EVENTS = {
    Event.CONNECTION_STARTED,
    Event.CONNECTION_FAILED,
    Event.CONNECTION_FINISHED,
}

SESSION_AND_DATA_EVENTS = {
    Event.START_SESSION,
    Event.CANCEL_SESSION,
    Event.FINISH_SESSION,
    Event.SESSION_STARTED,
    Event.SESSION_CANCELLED,
    Event.SESSION_FINISHED,
    Event.SESSION_FAILED,
    Event.TASK_REQUEST,
    Event.TTS_SENTENCE_START,
    Event.TTS_SENTENCE_END,
    Event.TTS_RESPONSE,
    Event.TTS_SUBTITLE,
}


class DoubaoProtocolError(RuntimeError):
    """豆包双向流式协议错误。"""


@dataclass
class ParsedFrame:
    message_type: int
    flags: int
    serialization: int
    compression: int
    event: int
    connection_id: str
    session_id: str
    payload: bytes
    json_payload: Any
    error_code: Optional[int] = None
    sequence_number: Optional[int] = None

    @property
    def is_error(self) -> bool:
        return self.message_type == MessageType.ERROR_INFORMATION

    @property
    def code(self) -> Optional[int]:
        if self.error_code is not None:
            return self.error_code
        if isinstance(self.json_payload, dict):
            for key in ("status_code", "code"):
                value = self.json_payload.get(key)
                if isinstance(value, int):
                    return value
        return None

    @property
    def message(self) -> str:
        if isinstance(self.json_payload, dict):
            for key in ("message", "error"):
                value = self.json_payload.get(key)
                if isinstance(value, str):
                    return value
        return ""

    def audio_bytes(self) -> bytes:
        if self.message_type in (
            MessageType.AUDIO_ONLY_REQUEST,
            MessageType.AUDIO_ONLY_RESPONSE,
        ):
            return self.payload

        if isinstance(self.json_payload, dict):
            data = self.json_payload.get("data")
            if isinstance(data, str):
                try:
                    return base64.b64decode(data)
                except Exception:
                    return b""
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)

        if self.serialization == Serialization.RAW:
            return self.payload

        return b""


def build_headers(
    credentials: DoubaoConfig, request_id: str, resource_id: Optional[str] = None
) -> Dict[str, str]:
    headers = {
        "X-Api-Access-Key": credentials.access_key,
        "X-Api-Connect-Id": request_id,
        "X-Api-Resource-Id": resource_id or credentials.resource_id,
        "X-Api-Request-Id": request_id,
    }
    if credentials.request_usage_tokens:
        headers["X-Control-Require-Usage-Tokens-Return"] = "*"
    return headers


def build_header_candidates(
    credentials: DoubaoConfig,
    request_id: str,
    resource_id: Optional[str] = None,
) -> List[Tuple[str, Dict[str, str]]]:
    base_headers = build_headers(credentials, request_id, resource_id=resource_id)
    candidates = []

    if credentials.send_app_key_header:
        headers = dict(base_headers)
        headers["X-Api-App-Key"] = credentials.app_id
        candidates.append(("app_key", headers))

    if credentials.send_app_id_header:
        headers = dict(base_headers)
        headers["X-Api-App-Id"] = credentials.app_id
        candidates.append(("app_id", headers))

    if credentials.send_app_id_header and credentials.send_app_key_header:
        headers = dict(base_headers)
        headers["X-Api-App-Id"] = credentials.app_id
        headers["X-Api-App-Key"] = credentials.app_id
        candidates.append(("both", headers))

    if not candidates:
        candidates.append(("base", dict(base_headers)))

    return candidates


def build_start_payload(settings: Any) -> Dict[str, Any]:
    req_params = _build_req_params(settings)
    return {
        "event": int(Event.START_SESSION),
        "namespace": "BidirectionalTTS",
        "user": {"uid": settings.uid},
        "req_params": req_params,
    }


def build_text_payload(text: str, settings: Any) -> Dict[str, Any]:
    req_params = _build_req_params(settings, text=text)
    return {
        "event": int(Event.TASK_REQUEST),
        "namespace": "BidirectionalTTS",
        "user": {"uid": settings.uid},
        "req_params": req_params,
    }


def _build_req_params(settings: Any, text: Optional[str] = None) -> Dict[str, Any]:
    audio_params = {
        "format": settings.audio_format,
        "sample_rate": settings.sample_rate,
        "speech_rate": settings.speech_rate,
        "loudness_rate": settings.loudness_rate,
    }
    if settings.emotion:
        audio_params["emotion"] = settings.emotion
        audio_params["emotion_scale"] = settings.emotion_scale
    if settings.enable_subtitle:
        audio_params["enable_subtitle"] = True
    if settings.enable_timestamp:
        audio_params["enable_timestamp"] = True

    additions = {
        "disable_markdown_filter": settings.disable_markdown_filter,
    }
    if settings.silence_duration_ms:
        additions["silence_duration"] = settings.silence_duration_ms
    if settings.explicit_language:
        additions["explicit_language"] = settings.explicit_language

    req_params = {
        "speaker": settings.speaker,
        "audio_params": audio_params,
        "additions": json.dumps(additions, ensure_ascii=False, separators=(",", ":")),
    }
    if text is not None:
        req_params["text"] = text
    if settings.model:
        req_params["model"] = settings.model
    return req_params


def build_finish_payload() -> Dict[str, Any]:
    return {}


def build_start_connection_frame() -> bytes:
    return _build_request_frame(Event.START_CONNECTION, _encode_json({}))


def build_finish_connection_frame() -> bytes:
    return _build_request_frame(Event.FINISH_CONNECTION, _encode_json({}))


def build_start_session_frame(session_id: str, payload: Dict[str, Any]) -> bytes:
    return _build_request_frame(Event.START_SESSION, _encode_json(payload), session_id=session_id)


def build_task_request_frame(session_id: str, payload: Dict[str, Any]) -> bytes:
    return _build_request_frame(Event.TASK_REQUEST, _encode_json(payload), session_id=session_id)


def build_finish_session_frame(session_id: str) -> bytes:
    return _build_request_frame(
        Event.FINISH_SESSION,
        _encode_json(build_finish_payload()),
        session_id=session_id,
    )


def build_cancel_session_frame(session_id: str, reason: str) -> bytes:
    del reason
    return _build_request_frame(
        Event.CANCEL_SESSION,
        _encode_json(build_finish_payload()),
        session_id=session_id,
    )


def parse_frame(frame: bytes) -> ParsedFrame:
    if len(frame) < HEADER_SIZE:
        raise DoubaoProtocolError("Frame too short")

    version = (frame[0] >> 4) & 0x0F
    header_words = frame[0] & 0x0F
    header_size = header_words * 4
    if version != PROTOCOL_VERSION:
        raise DoubaoProtocolError("Unsupported protocol version: %s" % version)
    if header_size < HEADER_SIZE or len(frame) < header_size:
        raise DoubaoProtocolError("Invalid header size")

    message_type = (frame[1] >> 4) & 0x0F
    flags = frame[1] & 0x0F
    serialization = (frame[2] >> 4) & 0x0F
    compression = frame[2] & 0x0F
    offset = header_size

    if message_type == MessageType.ERROR_INFORMATION:
        error_code = _read_uint32(frame, offset, "error_code")
        offset += 4
        payload = _read_error_payload(frame, offset)
        payload = _decode_compression(payload, compression)
        json_payload = _decode_payload(payload, serialization)
        return ParsedFrame(
            message_type=message_type,
            flags=flags,
            serialization=serialization,
            compression=compression,
            event=0,
            connection_id="",
            session_id="",
            payload=payload,
            json_payload=json_payload,
            error_code=error_code,
        )

    if message_type == MessageType.AUDIO_ONLY_RESPONSE:
        event = 0
        if flags & FLAG_WITH_EVENT:
            event = _read_uint32(frame, offset, "event")
            offset += 4

        session_id = ""
        sequence_number = None
        payload, session_id, sequence_number = _read_audio_only_payload(frame, offset)
        payload = _decode_compression(payload, compression)
        json_payload = _decode_payload(payload, serialization)
        return ParsedFrame(
            message_type=message_type,
            flags=flags,
            serialization=serialization,
            compression=compression,
            event=event or int(Event.TTS_RESPONSE),
            connection_id="",
            session_id=session_id,
            payload=payload,
            json_payload=json_payload,
            sequence_number=sequence_number,
        )

    if message_type == MessageType.FRONTEND_SERVER_RESPONSE:
        payload, _ = _read_length_prefixed_payload(frame, offset)
        payload = _decode_compression(payload, compression)
        json_payload = _decode_payload(payload, serialization)
        return ParsedFrame(
            message_type=message_type,
            flags=flags,
            serialization=serialization,
            compression=compression,
            event=0,
            connection_id="",
            session_id="",
            payload=payload,
            json_payload=json_payload,
        )

    event = 0
    if flags & FLAG_WITH_EVENT:
        event = _read_uint32(frame, offset, "event")
        offset += 4

    connection_id = ""
    session_id = ""
    identifier_kind = _identifier_kind(message_type, event)
    if identifier_kind == "connection":
        connection_id, payload, offset = _read_identifier_and_payload(
            frame,
            offset,
            strict_identifier=False,
            label="connection_id",
        )
    elif identifier_kind == "session":
        session_id, payload, offset = _read_identifier_and_payload(
            frame,
            offset,
            strict_identifier=True,
            label="session_id",
        )
    else:
        payload, offset = _read_length_prefixed_payload(frame, offset)

    payload = _decode_compression(payload, compression)
    json_payload = _decode_payload(payload, serialization)
    return ParsedFrame(
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        event=event,
        connection_id=connection_id,
        session_id=session_id,
        payload=payload,
        json_payload=json_payload,
    )


def _encode_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _build_request_frame(event: Event, payload: bytes, session_id: str = "") -> bytes:
    frame = bytearray()
    frame.append((PROTOCOL_VERSION << 4) | HEADER_WORDS)
    frame.append((MessageType.FULL_CLIENT_REQUEST << 4) | FLAG_WITH_EVENT)
    frame.append((Serialization.JSON << 4) | Compression.NONE)
    frame.append(0)
    frame.extend(int(event).to_bytes(4, "big"))

    if session_id:
        session_bytes = session_id.encode("utf-8")
        frame.extend(len(session_bytes).to_bytes(4, "big"))
        frame.extend(session_bytes)

    frame.extend(len(payload).to_bytes(4, "big"))
    frame.extend(payload)
    return bytes(frame)


def _identifier_kind(message_type: int, event: int) -> Optional[str]:
    if event == 0:
        return None

    try:
        event_enum = Event(event)
    except ValueError:
        return None

    if message_type == MessageType.FULL_CLIENT_REQUEST:
        if event_enum in CONNECT_REQUEST_EVENTS:
            return None
        if event_enum in SESSION_AND_DATA_EVENTS:
            return "session"
        return None

    if message_type == MessageType.FULL_SERVER_RESPONSE:
        if event_enum in CONNECT_RESPONSE_EVENTS:
            return "connection"
        if event_enum in SESSION_AND_DATA_EVENTS:
            return "session"
        return None

    return None


def _read_audio_only_payload(
    frame: bytes,
    offset: int,
) -> Tuple[bytes, str, Optional[int]]:
    if len(frame) == offset:
        return b"", "", None

    if len(frame) == offset + 4 and frame[offset : offset + 4] == b"\x00\x00\x00\x00":
        return b"", "", None

    try:
        session_id, payload, _ = _read_identifier_and_payload(
            frame,
            offset,
            strict_identifier=True,
            label="session_id",
        )
        return payload, session_id, None
    except Exception:
        pass

    if len(frame) >= offset + 8:
        sequence_number = int.from_bytes(frame[offset : offset + 4], "big", signed=True)
        payload, _ = _read_length_prefixed_payload(frame, offset + 4)
        return payload, "", sequence_number

    payload, _ = _read_length_prefixed_payload(frame, offset)
    return payload, "", None


def _read_identifier_and_payload(
    frame: bytes,
    offset: int,
    *,
    strict_identifier: bool,
    label: str,
) -> Tuple[str, bytes, int]:
    identifier_length = _read_uint32(frame, offset, "%s_length" % label)
    identifier_start = offset + 4
    identifier_end = identifier_start + identifier_length

    if len(frame) >= identifier_end + 4:
        payload_length = int.from_bytes(frame[identifier_end : identifier_end + 4], "big")
        payload_end = identifier_end + 4 + payload_length
        if payload_end == len(frame):
            identifier = frame[identifier_start:identifier_end].decode("utf-8")
            payload = frame[identifier_end + 4 : payload_end]
            return identifier, payload, payload_end

    if strict_identifier:
        raise DoubaoProtocolError("Invalid %s field layout" % label)

    payload, payload_end = _read_length_prefixed_payload(frame, offset)
    return "", payload, payload_end


def _read_length_prefixed_payload(frame: bytes, offset: int) -> Tuple[bytes, int]:
    payload_length = _read_uint32(frame, offset, "payload_length")
    payload_start = offset + 4
    payload_end = payload_start + payload_length
    if payload_end > len(frame):
        raise DoubaoProtocolError(
            "Declared payload length exceeds frame size: payload_end=%s, frame_len=%s"
            % (payload_end, len(frame))
        )
    if payload_end != len(frame):
        raise DoubaoProtocolError(
            "Declared payload length does not match frame size: payload_end=%s, frame_len=%s"
            % (payload_end, len(frame))
        )
    return frame[payload_start:payload_end], payload_end


def _read_error_payload(frame: bytes, offset: int) -> bytes:
    if len(frame) >= offset + 4:
        payload_length = int.from_bytes(frame[offset : offset + 4], "big")
        payload_start = offset + 4
        payload_end = payload_start + payload_length
        if payload_end == len(frame):
            return frame[payload_start:payload_end]
    return frame[offset:]


def _read_uint32(frame: bytes, offset: int, field_name: str) -> int:
    end = offset + 4
    if end > len(frame):
        raise DoubaoProtocolError("Missing %s" % field_name)
    return int.from_bytes(frame[offset:end], "big")


def _decode_compression(payload: bytes, compression: int) -> bytes:
    if not payload:
        return payload
    if compression == Compression.NONE:
        return payload
    if compression == Compression.GZIP:
        return gzip.decompress(payload)
    raise DoubaoProtocolError("Unsupported compression method: %s" % compression)


def _decode_payload(payload: bytes, serialization: int) -> Any:
    if not payload:
        return None
    if serialization == Serialization.RAW:
        return None
    if serialization != Serialization.JSON:
        raise DoubaoProtocolError("Unsupported serialization method: %s" % serialization)
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
