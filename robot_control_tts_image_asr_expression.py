# -*- coding: utf-8 -*-
"""
Standalone robot dog control program with:
- text input and ASR input
- robot movement through ROS /cmd_vel
- camera scene description tool
- Qwen realtime TTS
- emotion classification
- expression image display handoff
"""

import argparse
import base64
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time

import cv2
import dashscope
import pyaudio
from dashscope.audio.qwen_omni import (
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from openai import OpenAI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

EXPRESSION_DISPLAY_SCRIPT = os.path.join(SCRIPT_DIR, "expression_image_framebuffer.py")
VOICE_TRIGGERS = {"v", "voice", "语音"}

ROS_AVAILABLE = False
_cmd_vel_pub = None


def init_ros():
    global ROS_AVAILABLE, _cmd_vel_pub
    try:
        import rospy
        from geometry_msgs.msg import Twist  # noqa: F401
        if not rospy.core.is_initialized():
            rospy.init_node("robot_dog_control", anonymous=True)
            time.sleep(0.2)
        _cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        time.sleep(0.3)
        ROS_AVAILABLE = True
        print("[ROS] OK /cmd_vel ready")
        return True
    except ImportError:
        print("[ROS] rospy not found. Please install ROS and rospy.")
    except Exception as e:
        print("[ROS] init failed: {}".format(e))
    return False


def publish_cmd_vel(linear, angular):
    global _cmd_vel_pub
    if _cmd_vel_pub is None:
        return False
    try:
        from geometry_msgs.msg import Twist
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        _cmd_vel_pub.publish(twist)
        return True
    except Exception as e:
        print("[ROS] publish failed: {}".format(e))
        return False


VOICE_MAP = {
    "zh": "Cherry",
    "en": "Stella",
}

EMOTION_LABELS = {
    "neutral",
    "happy",
    "thinking",
    "warning",
    "sad",
    "goodbye",
    "moving",
}

EMOTION_EMOJI = {
    "neutral": "\U0001F642",
    "happy": "\U0001F604",
    "thinking": "\U0001F914",
    "warning": "\u26A0\uFE0F",
    "sad": "\U0001F61F",
    "goodbye": "\U0001F44B",
    "moving": "\U0001F6B6",
}

EMOTION_CLASSIFIER_PROMPT = (
    "你是机器狗助手的表情分类器。"
    "根据机器人即将说出的当前情境以及用户的言论，从 neutral, happy, thinking, warning, sad, goodbye, moving 中选择一个最合适的标签。"
    "规则："
    "- 用户夸奖机器狗聪明可爱时 → happy"
    "- 用户说机器狗笨时 → sad"
    "- 执行移动命令（前进、转圈、倒车等）时 → neutral"
    "- 收到无法完成的指令（如飞起来）并警告用户时 → warning"
    "- 与用户道别说再见时 → goodbye"
    "- 调用 describe_scene 思考并描述画面时 → thinking"
    "- 普通聊天无特殊情感 → neutral"
    "不要解释，不要输出多余内容，只输出一个标签。"
)


class _SpeakCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        self.complete_event = threading.Event()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
        )

    def on_open(self):
        print("[TTS] connection opened")

    def on_close(self, close_status_code, close_msg):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("[TTS] connection closed (code: {}, msg: {})".format(close_status_code, close_msg))

    def on_event(self, response):
        try:
            typ = response.get("type", "")
            if typ == "session.created":
                print("[TTS] session started: {}".format(response["session"]["id"]))
            elif typ == "response.audio.delta":
                audio_bytes = base64.b64decode(response["delta"])
                self.stream.write(audio_bytes)
            elif typ == "response.done":
                print("[TTS] response done")
            elif typ == "session.finished":
                print("[TTS] session finished")
                self.complete_event.set()
        except Exception as e:
            print("[TTS Error] {}".format(e))

    def wait_for_finished(self):
        self.complete_event.wait()


def init_tts_api_key():
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        dashscope.api_key = "sk-eb97b5349dee49ffa261d418dc95a459"


def speak_text(text, voice="zh"):
    if not text:
        return
    voice_name = VOICE_MAP.get(voice, "Cherry")
    try:
        callback = _SpeakCallback()
        tts = QwenTtsRealtime(
            model="qwen3-tts-flash-realtime",
            callback=callback,
            url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        )
        tts.connect()
        tts.update_session(
            voice=voice_name,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
        )
        tts.append_text(text)
        tts.finish()
        callback.wait_for_finished()
    except Exception as e:
        print("TTS failed: {}".format(e))


def normalize_emotion(label):
    normalized = str(label or "").strip().lower()
    return normalized if normalized in EMOTION_LABELS else "neutral"


def get_expression_emoji(emotion):
    return EMOTION_EMOJI[normalize_emotion(emotion)]


def classify_emotion(emotion_client, reply_text, model="deepseek-chat", result_type=None, command_data=None):
    fallback_emotion = "moving" if result_type == "command" and command_data else "neutral"
    if not reply_text or emotion_client is None:
        return fallback_emotion

    try:
        response = emotion_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EMOTION_CLASSIFIER_PROMPT},
                {"role": "user", "content": reply_text},
            ],
            stream=False,
        )
        return normalize_emotion(response.choices[0].message.content) or fallback_emotion
    except Exception as e:
        print("[EMOTION] classify failed: {}".format(e))
        return fallback_emotion


def send_expression_to_display(emotion):
    normalized = normalize_emotion(emotion)
    try:
        result = subprocess.run(
            [sys.executable, EXPRESSION_DISPLAY_SCRIPT, normalized, "--no-init"],
            check=False,
        )
        if result.returncode != 0:
            print("[LED] display script failed: {}".format(result.returncode))
            return False
        return True
    except Exception as e:
        print("[LED] display script error: {}".format(e))
        return False


def show_expression(emotion):
    normalized = normalize_emotion(emotion)
    send_expression_to_display(normalized)
    print("[LED] show: {}".format(normalized))


def speak_and_show(reply_text, emotion, voice="zh"):
    show_expression(normalize_emotion(emotion))
    speak_text(reply_text, voice=voice)


def get_rms(data):
    count = len(data) // 2
    if count <= 0:
        return 0.0
    shorts = struct.unpack("{}h".format(count), data)
    sum_squares = sum(sample ** 2 for sample in shorts)
    return (sum_squares / float(count)) ** 0.5


class _ASRCallback(OmniRealtimeCallback):
    def __init__(self):
        self.transcript = ""
        self.stop_event = threading.Event()
        self.has_speech_started = False
        self.last_activity_time = time.time()
        self.handlers = {
            "session.created": self._handle_session_created,
            "conversation.item.input_audio_transcription.completed": self._handle_final_text,
            "conversation.item.input_audio_transcription.text": self._handle_transcription_text,
            "input_audio_buffer.speech_started": self._handle_speech_start,
            "input_audio_buffer.speech_stopped": self._handle_speech_stop,
        }

    def on_open(self):
        print("[ASR] connection opened")

    def on_close(self, code, msg):
        print("[ASR] connection closed, code: {}, msg: {}".format(code, msg))

    def on_event(self, response):
        try:
            handler = self.handlers.get(response.get("type"))
            if handler:
                handler(response)
        except Exception as e:
            print("[ASR] callback error: {}".format(e))

    def _handle_session_created(self, response):
        session = response.get("session", {})
        print("[ASR] session started: {}".format(session.get("id", "")))

    def _handle_final_text(self, response):
        self.transcript = response.get("transcript", "")
        self.last_activity_time = time.time()
        print("[ASR] final text: {}".format(self.transcript))

    def _handle_transcription_text(self, response):
        text = response.get("text", "")
        stash = response.get("stash", "")
        print("[ASR] partial text: {}".format(text + stash))
        self.last_activity_time = time.time()

    def _handle_speech_start(self, response):
        print("[ASR] speech started")
        self.has_speech_started = True
        self.last_activity_time = time.time()

    def _handle_speech_stop(self, response):
        print("[ASR] speech stopped")
        self.last_activity_time = time.time()
        self.stop_event.set()


class VoiceRecognizer:
    def __init__(self, api_key, model="qwen3-asr-flash-realtime", url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime"):
        dashscope.api_key = api_key
        self.model = model
        self.url = url
        logging.getLogger("dashscope").setLevel(logging.WARNING)

    def listen(
        self,
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=100,
        max_duration=10.0,
        idle_timeout=3.0,
        enable_vad=True,
        silence_threshold=500,
        silence_chunks=30,
        input_device_index=29,
    ):
        callback = _ASRCallback()
        conversation = OmniRealtimeConversation(
            model=self.model,
            url=self.url,
            callback=callback,
        )
        try:
            conversation.connect()
        except Exception as e:
            print("[ASR] connect failed: {}".format(e))
            return ""

        from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

        transcription_params = TranscriptionParams(
            language="zh",
            sample_rate=sample_rate,
            input_audio_format="pcm",
        )
        conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params,
        )

        chunk = int(sample_rate * chunk_duration_ms / 1000)
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk,
            input_device_index=input_device_index,
        )

        print("[ASR] microphone started")
        start_time = time.time()
        silent_count = 0
        try:
            while not callback.stop_event.is_set():
                if time.time() - start_time > max_duration:
                    print("[ASR] max duration reached")
                    break
                if not callback.has_speech_started and time.time() - callback.last_activity_time > idle_timeout:
                    print("[ASR] idle timeout")
                    break
                if enable_vad and silent_count > silence_chunks:
                    print("[ASR] local silence timeout")
                    break
                data = stream.read(chunk, exception_on_overflow=False)
                if enable_vad:
                    if get_rms(data) < silence_threshold:
                        silent_count += 1
                    else:
                        silent_count = 0
                conversation.append_audio(base64.b64encode(data).decode("ascii"))
        except KeyboardInterrupt:
            print("\n[ASR] interrupted")
        except Exception as e:
            print("[ASR] recording error: {}".format(e))
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            try:
                conversation.end_session()
            except Exception as e:
                print("[ASR] end_session ignored: {}".format(e))
            try:
                conversation.close()
            except Exception as e:
                print("[ASR] close ignored: {}".format(e))

        return callback.transcript.strip()


MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move_robot",
        "description": "控制机器狗移动，直接指定线速度、角速度和持续时间",
        "parameters": {
            "type": "object",
            "properties": {
                "linear_velocity": {"type": "number", "description": "线速度 m/s，正值前进，负值后退"},
                "radian_velocity": {"type": "number", "description": "角速度 rad/s，正值左转，负值右转"},
                "duration_seconds": {"type": "number", "description": "持续时间，默认 2 秒"},
            },
            "required": ["linear_velocity", "radian_velocity"],
        },
    },
}

DESCRIBE_TOOL = {
    "type": "function",
    "function": {
        "name": "describe_scene",
        "description": "使用摄像头和视觉模型描述当前画面，用于回答你看到了什么。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

AVAILABLE_TOOLS = [MOVE_TOOL, DESCRIBE_TOOL]


def execute_move(linear, angular, duration=2.0):
    cmd_json = {
        "timestamp": time.time(),
        "command": "move",
        "linear_velocity": round(linear, 2),
        "radian_velocity": round(angular, 2),
        "duration_seconds": duration,
    }
    print("\n{}\n[CMD_JSON] {}\n{}".format("=" * 50, json.dumps(cmd_json, ensure_ascii=False), "=" * 50))

    if linear == 0.0 and angular == 0.0:
        publish_cmd_vel(0.0, 0.0)
        print("[CTRL] STOP")
        time.sleep(0.1)
        return json.dumps({"status": "stopped"})

    print("[CTRL] GO | linear={:.2f} m/s angular={:.2f} rad/s | duration={:.1f}s".format(linear, angular, duration))
    start = time.time()
    count = 0
    while time.time() - start < duration:
        publish_cmd_vel(linear, angular)
        count += 1
        time.sleep(0.05)
    publish_cmd_vel(0.0, 0.0)
    print("[CTRL] DONE! sent {} times, elapsed {:.2f}s".format(count, time.time() - start))
    return json.dumps({"status": "success", "linear": linear, "angular": angular, "duration": duration})


def capture_and_encode_frame(camera_id=0, max_tries=30, pre_warm=5):
    from GStreamerWrapper.GStreamerWrapper import GStreamerWrapper

    gsw = GStreamerWrapper()
    try:
        for _ in range(pre_warm):
            gsw.GetFrame()
            time.sleep(0.05)
        frame = None
        for _ in range(max_tries):
            frame = gsw.GetFrame()
            if frame is not None:
                break
            time.sleep(0.1)
        if frame is None:
            raise IOError("camera frame unavailable after retries")
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return "data:image/jpeg;base64,{}".format(jpg_as_text)
    finally:
        gsw.StopThread()


def describe_scene(vision_client, vision_model="qwen3.6-plus"):
    image_url = capture_and_encode_frame()
    response = vision_client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请用简洁的中文描述图中景象，包括主要物体、人物、位置关系等。"},
                ],
            }
        ],
    )
    desc = response.choices[0].message.content
    print("[VISION] {}".format(desc))
    return desc


class RobotDogAgent:
    def __init__(self, chat_api_key, chat_base_url, chat_model, vision_api_key, vision_base_url, vision_model):
        self.chat_client = OpenAI(api_key=chat_api_key, base_url=chat_base_url)
        self.chat_model = chat_model
        self.vision_client = OpenAI(api_key=vision_api_key, base_url=vision_base_url)
        self.vision_model = vision_model
        self.system_prompt = (
            "你是一个聪明的机器狗助手，能理解中文指令。\n"
            "当用户想要移动机器狗时，包括走快一点、慢慢转圈、倒车等模糊说法，"
            "请调用 move_robot 工具，直接给出合适的线速度、角速度和持续时间。\n"
            "线速度正值为前进，负值为后退；角速度正值为左转，负值为右转。\n"
            "当用户询问你看到了什么或类似问题时，请调用 describe_scene 工具获取当前画面描述，"
            "然后基于描述回答用户，这时你感受到在思考。\n"
            "如果只是普通聊天，请直接回复，不要调用工具。"
            "当用户说你笨时，你会感到伤心。\n"
            "当用户夸奖你聪明可爱时，你会感到开心。\n "
            "当你正在执行移动命令时，你会感到平静。\n"
            "当你被下达了明显无法完成的指令，你会警告用户。\n"
            "当你要和用户道别时，你会感到依依不舍并说再见。\n"
            
        )

    def process(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        try:
            resp = self.chat_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                stream=False,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return {"type": "chat", "reply": msg.content or "", "command_data": None}

            messages.append(msg)
            command_info = None
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_result = None
                if tool_name == "move_robot":
                    lin = args.get("linear_velocity", 0.0)
                    ang = args.get("radian_velocity", 0.0)
                    dur = args.get("duration_seconds", 2.0)
                    tool_result = execute_move(lin, ang, dur)
                    command_info = {"linear_velocity": lin, "radian_velocity": ang, "duration_seconds": dur}
                elif tool_name == "describe_scene":
                    tool_result = describe_scene(self.vision_client, self.vision_model)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(tool_result)})

            resp2 = self.chat_client.chat.completions.create(model=self.chat_model, messages=messages, stream=False)
            final_reply = resp2.choices[0].message.content or ""
            if command_info:
                return {"type": "command", "reply": final_reply, "command_data": command_info}
            return {"type": "chat", "reply": final_reply, "command_data": None}
        except Exception as e:
            return {"type": "error", "reply": "API error: {}".format(e), "command_data": None}


def is_voice_trigger(text):
    return str(text or "").strip().lower() in VOICE_TRIGGERS


def resolve_user_input(text, recognizer, input_device_index=29):
    stripped = str(text or "").strip()
    if not is_voice_trigger(stripped):
        return stripped
    print("[ASR] voice input mode, please speak...")
    spoken_text = recognizer.listen(max_duration=10.0, idle_timeout=3.0, input_device_index=input_device_index)
    if spoken_text:
        print("[ASR] recognized: {}".format(spoken_text))
    else:
        print("[ASR] no valid speech recognized")
    return spoken_text


def handle_result(result, agent, voice):
    if result is None:
        print("Robot: ? Something went wrong")
        speak_and_show("请重试", "sad", voice=voice)
        return

    if result["type"] == "command":
        print("Robot: " + result["reply"])
        command_data = result["command_data"]
        print("      |- 线速度: {:.2f} m/s".format(command_data["linear_velocity"]))
        print("      |- 角速度: {:.2f} rad/s".format(command_data["radian_velocity"]))
        print("      |- 持续时间: {:.1f} s".format(command_data["duration_seconds"]))
        emotion = classify_emotion(
            agent.chat_client,
            result["reply"],
            model=agent.chat_model,
            result_type=result["type"],
            command_data=command_data,
        )
        speak_and_show(result["reply"], emotion, voice=voice)
        return

    if result["type"] == "chat":
        print("Robot: " + result["reply"])
        emotion = classify_emotion(agent.chat_client, result["reply"], model=agent.chat_model, result_type=result["type"])
        speak_and_show(result["reply"], emotion, voice=voice)
        return

    print("Error: " + result["reply"])
    emotion = classify_emotion(agent.chat_client, result["reply"], model=agent.chat_model, result_type=result["type"])
    speak_and_show(result["reply"], emotion, voice=voice)


def main():
    parser = argparse.ArgumentParser(description="Robot Dog Control (standalone ASR + TTS + Vision + Expression Images)")
    parser.add_argument("--chat_api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--chat_base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--chat_model", type=str, default="deepseek-chat")
    parser.add_argument("--vision_api_key", type=str, default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--vision_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--vision_model", type=str, default="qwen3.6-plus")
    parser.add_argument("--voice", type=str, default="zh", choices=["zh", "en"])
    parser.add_argument("--input_device_index", type=int, default=29)
    args = parser.parse_args()

    if not init_ros():
        print("FATAL: Cannot initialize ROS /cmd_vel publisher. Exiting.")
        sys.exit(1)

    init_tts_api_key()
    recognizer = VoiceRecognizer(api_key=args.vision_api_key)
    try:
        agent = RobotDogAgent(
            chat_api_key=args.chat_api_key,
            chat_base_url=args.chat_base_url,
            chat_model=args.chat_model,
            vision_api_key=args.vision_api_key,
            vision_base_url=args.vision_base_url,
            vision_model=args.vision_model,
        )
    except Exception as e:
        print("Agent init failed: {}".format(e))
        sys.exit(1)

    print("=" * 55)
    print("  Robot Dog Control (standalone ASR + TTS + Vision + Expression)")
    print("  Input text directly, or enter v / voice / 语音 for voice input")
    print("  Exit: exit / quit / q")
    print("=" * 55)

    while True:
        try:
            text = input("\nYou (text or v): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            speak_and_show("再见!", "goodbye", voice=args.voice)
            break

        if text.lower() in ["exit", "quit", "q"]:
            print("Bye!")
            speak_and_show("再见!", "goodbye", voice=args.voice)
            break

        text = resolve_user_input(text, recognizer, input_device_index=args.input_device_index)
        if not text:
            continue

        result = agent.process(text)
        handle_result(result, agent, args.voice)


if __name__ == "__main__":
    main()
