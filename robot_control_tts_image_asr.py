# robot_control_tts_image.py (整合语音识别版)
# -*- coding: utf-8 -*-
"""
机器狗智能控制程序 v5.0 —— 视觉 + 语音输入 + 真实环境 + 阿里云 Qwen 实时 TTS
新增功能：集成阿里云 Qwen Omni 实时语音识别，支持语音命令输入。
使用方法：
    - 直接输入文本命令（支持自然语言）
    - 输入 'v' 或 '语音' 进入语音输入模式，说出命令后自动识别并执行
"""

import os
import json
import argparse
import time
import sys
import threading
import base64
import struct
import logging

import cv2
import pyaudio
import dashscope
from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation,
    OmniRealtimeCallback,
    MultiModality,        # 保留 MultiModality（如果没有也换成字符串 "text"）
)
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
    AudioFormat,
)
from openai import OpenAI

# ============================================================
# 0. ROS 直接控制（始终真实）
# ============================================================
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: {dev['name']}")
p.terminate()
ROS_AVAILABLE = False
_cmd_vel_pub = None

def init_ros():
    """初始化 ROS 并创建 /cmd_vel 发布器"""
    global ROS_AVAILABLE, _cmd_vel_pub
    try:
        import rospy
        from geometry_msgs.msg import Twist
        if not rospy.core.is_initialized():
            rospy.init_node('robot_dog_control', anonymous=True)
            time.sleep(0.2)
        _cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        time.sleep(0.3)
        ROS_AVAILABLE = True
        print("[ROS] OK /cmd_vel ready")
        return True
    except ImportError:
        print("[ROS] rospy not found. Please install ROS and rospy.")
    except Exception as e:
        print("[ROS] init failed: %s" % str(e))
    return False

def publish_cmd_vel(linear, angular):
    """发布速度命令到 /cmd_vel"""
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
        print("[ROS] publish failed: %s" % str(e))
        return False

# ============================================================
# 0.5 语音合成（阿里云 Qwen TTS Realtime）
# ============================================================

VOICE_MAP = {
    "zh": "Cherry",
    "en": "Stella",
}

class _SpeakCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        self.complete_event = threading.Event()
        self.p = pyaudio.PyAudio()
        self.SAMPLE_RATE = 24000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            output=True,
            
        )

    def on_open(self) -> None:
        print('[TTS] connection opened')

    def on_close(self, close_status_code, close_msg) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print('[TTS] connection closed (code: {}, msg: {})'.format(close_status_code, close_msg))

    def on_event(self, response: str) -> None:
        try:
            typ = response.get('type', '')
            if typ == 'session.created':
                print('[TTS] session started: {}'.format(response['session']['id']))
            elif typ == 'response.audio.delta':
                recv_audio_b64 = response['delta']
                audio_bytes = base64.b64decode(recv_audio_b64)
                self.stream.write(audio_bytes)
            elif typ == 'response.done':
                print('[TTS] response done')
            elif typ == 'session.finished':
                print('[TTS] session finished')
                self.complete_event.set()
        except Exception as e:
            print('[TTS Error] {}'.format(e))

    def wait_for_finished(self):
        self.complete_event.wait()

def init_tts_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
    else:
        dashscope.api_key = 'sk-eb97b5349dee49ffa261d418dc95a459'  # 建议改用环境变量

def speak_text(text, voice="zh"):
    if not text:
        return
    voice_name = VOICE_MAP.get(voice, "Cherry")
    try:
        callback = _SpeakCallback()
        tts = QwenTtsRealtime(
            model='qwen3-tts-flash-realtime',
            callback=callback,
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
        )
        tts.connect()
        tts.update_session(
            voice=voice_name,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode='server_commit'
        )
        tts.append_text(text)
        tts.finish()
        callback.wait_for_finished()
    except Exception as e:
        print("❌ TTS 朗读失败: %s" % str(e))

# ============================================================
# 语音识别（阿里云 Qwen Omni 实时转录）
# ============================================================

def get_rms(data):
    """计算 PCM 音频块的均方根（能量），用于音量检测"""
    count = len(data) // 2
    shorts = struct.unpack(f"{count}h", data)
    sum_squares = sum(s ** 2 for s in shorts)
    return (sum_squares / count) ** 0.5


class _ASRCallback(OmniRealtimeCallback):
    """语音识别回调，提取最终转录文本并控制录音停止"""
    def __init__(self):
        self.transcript = ""
        self.stop_event = threading.Event()
        self.has_speech_started = False
        self.last_activity_time = time.time()

        self.handlers = {
            'session.created': self._handle_session_created,
            'conversation.item.input_audio_transcription.completed': self._handle_final_text,
            'conversation.item.input_audio_transcription.text': self._handle_transcription_text,
            'input_audio_buffer.speech_started': self._handle_speech_start,
            'input_audio_buffer.speech_stopped': self._handle_speech_stop,
        }

    def on_open(self):
        print('Connection opened')

    def on_close(self, code, msg):
        print(f'Connection closed, code: {code}, msg: {msg}')

    def on_event(self, response):
        try:
            handler = self.handlers.get(response['type'])
            if handler:
                handler(response)
        except Exception as e:
            print(f'[Error] {e}')

    def _handle_session_created(self, response):
        print(f"Start session: {response['session']['id']}")

    def _handle_final_text(self, response):
        self.transcript = response['transcript']
        self.last_activity_time = time.time()
        print(f"✅ Final recognized text: {self.transcript}")

    def _handle_transcription_text(self, response):
        print(f"Got transcription result: {response['text'] + response['stash']}")
        self.last_activity_time = time.time()

    def _handle_speech_start(self, response):
        print("🔊 检测到语音...")
        self.has_speech_started = True
        self.last_activity_time = time.time()

    def _handle_speech_stop(self, response):
        print("🔇 语音结束")
        self.last_activity_time = time.time()
        self.stop_event.set()  # 触发停止录音

class VoiceRecognizer:
    """封装实时语音转写功能，返回识别的文本"""
    def __init__(self, api_key, model='qwen3-asr-flash-realtime',
                 url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'):
        dashscope.api_key = api_key
        self.model = model
        self.url = url
        # 关闭 dashscope 内部 DEBUG 日志，避免干扰主程序
        logging.getLogger('dashscope').setLevel(logging.WARNING)

    def listen(self,
               sample_rate=16000, channels=1,
               chunk_duration_ms=100,
               max_duration=60.0,
               idle_timeout=5.0,
               enable_vad=True,
               silence_threshold=500,
               silence_chunks=30,
               input_device_index=29):
        """
        开始麦克风录音并实时转写，返回识别到的文本。
        支持多种自动停止条件：语音结束事件、超时、空闲超时、客户端静音检测。

        :param sample_rate: 采样率 (16kHz)
        :param channels: 声道数 (1)
        :param chunk_duration_ms: 每次发送的音频时长 (ms)
        :param max_duration: 最大录制时长 (秒)，超时自动停止
        :param idle_timeout: 空闲超时 (秒)，无语音活动时自动停止
        :param enable_vad: 是否启用客户端能量检测
        :param silence_threshold: VAD 静音阈值
        :param silence_chunks: 连续静音块数达到此值时停止
        :param input_device_index: 麦克风设备索引，默认 PulseAudio 设备 (29)
        """
        callback = _ASRCallback()
        conversation = OmniRealtimeConversation(
            model=self.model,
            url=self.url,
            callback=callback
        )

        try:
            conversation.connect()
        except Exception as e:
            print(f"❌ 语音识别连接失败: {e}")
            return ""

        # 配置转录参数（使用 TranscriptionParams 方式，与程序1一致）
        from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
        transcription_params = TranscriptionParams(
            language='zh',
            sample_rate=sample_rate,
            input_audio_format="pcm"
        )
        conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params
        )

        # 打开麦克风（使用程序1的配置参数）
        FORMAT = pyaudio.paInt16
        CHUNK = int(sample_rate * chunk_duration_ms / 1000)
        chunk_bytes = CHUNK * channels * 2

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=input_device_index
        )

        print(f"🎤 麦克风已启动，将自动在以下条件触发时停止：")
        print(f"   - 检测到语音结束 (speech_stopped 事件)")
        print(f"   - 最大录制时长 {max_duration}s")
        print(f"   - 空闲超时 {idle_timeout}s")
        if enable_vad:
            print(f"   - 客户端静音检测：连续 {silence_chunks} 块 RMS < {silence_threshold}")
        print("按 Ctrl+C 可随时手动停止。\n")

        start_time = time.time()
        silent_count = 0

        try:
            while not callback.stop_event.is_set():
                # 1. 最大录制时长
                if time.time() - start_time > max_duration:
                    print(f"⏰ 已达到最大录制时长 {max_duration}s，停止。")
                    break

                # 2. 空闲超时：从未检测到语音开始，且长时间无任何活动
                if (not callback.has_speech_started and
                        time.time() - callback.last_activity_time > idle_timeout):
                    print(f"⏳ 超过 {idle_timeout}s 未检测到语音活动，自动停止。")
                    break

                # 3. (可选) 客户端 VAD：连续静音块数
                if enable_vad and silent_count > silence_chunks:
                    print(f"🔇 连续静音达到 {silence_chunks} 块，停止录制。")
                    break

                # 从麦克风读取一块音频
                data = stream.read(CHUNK, exception_on_overflow=False)

                # 客户端能量检测（可选）
                if enable_vad:
                    rms = get_rms(data)
                    if rms < silence_threshold:
                        silent_count += 1
                    else:
                        silent_count = 0
                else:
                    silent_count = 0

                # 编码并发送
                audio_b64 = base64.b64encode(data).decode('ascii')
                conversation.append_audio(audio_b64)

        except KeyboardInterrupt:
            print("\n⚠️ 用户手动中断 (Ctrl+C)。")
        except Exception as e:
            print(f"❌ 录音异常: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            try:
                conversation.end_session()
            except Exception as e:
                print(f"[ASR] 忽略结束会话错误: {e}")
            try:
                conversation.close()
            except Exception as e:
                print(f"[ASR] 忽略关闭连接错误: {e}")

        return callback.transcript.strip()

# ============================================================
# 1. 工具定义（Move + Vision）
# ============================================================

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move_robot",
        "description": "控制机器狗移动，直接指定线速度、角速度和持续时间",
        "parameters": {
            "type": "object",
            "properties": {
                "linear_velocity": {
                    "type": "number",
                    "description": "线速度 (m/s)，正值向前，负值后退，范围 -1.0 到 1.0",
                },
                "radian_velocity": {
                    "type": "number",
                    "description": "角速度 (rad/s)，正值左转，负值右转，范围 -1.2 到 1.2",
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "持续时间（秒），默认 2.0",
                },
            },
            "required": ["linear_velocity", "radian_velocity"],
        },
    },
}

DESCRIBE_TOOL = {
    "type": "function",
    "function": {
        "name": "describe_scene",
        "description": "用视觉模型查看当前摄像头画面，返回对眼前景象的中文描述。用于回答'你看到了什么'等问题。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
    },
}

AVAILABLE_TOOLS = [MOVE_TOOL, DESCRIBE_TOOL]

# ============================================================
# 2. 工具实现
# ============================================================

def execute_move(linear, angular, duration=2.0):
    """移动控制"""
    cmd_json = {
        "timestamp": time.time(),
        "command": "move",
        "linear_velocity": round(linear, 2),
        "radian_velocity": round(angular, 2),
        "duration_seconds": duration,
    }
    print(f"\n{'='*50}\n[CMD_JSON] {json.dumps(cmd_json, ensure_ascii=False)}\n{'='*50}")

    if linear == 0.0 and angular == 0.0:
        publish_cmd_vel(0.0, 0.0)
        print("[CTRL] STOP")
        time.sleep(0.1)
        return json.dumps({"status": "stopped"})

    print(f"[CTRL] GO | linear={linear:.2f} m/s angular={angular:.2f} rad/s | duration={duration:.1f}s")
    start = time.time()
    count = 0
    while time.time() - start < duration:
        publish_cmd_vel(linear, angular)
        count += 1
        time.sleep(0.05)
    publish_cmd_vel(0.0, 0.0)
    print(f"[CTRL] DONE! sent {count} times, elapsed {time.time()-start:.2f}s")
    return json.dumps({
        "status": "success",
        "linear": linear,
        "angular": angular,
        "duration": duration
    })

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
            raise IOError("多次尝试后仍无法从摄像头获取图像")
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"
    finally:
        gsw.StopThread()

def describe_scene(vision_client, vision_model="qwen3.6-plus"):
    """调用视觉模型描述当前画面"""
    image_url = capture_and_encode_frame()
    response = vision_client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "请用简洁的中文描述图中描绘的景象，包括主要物体、人物、位置关系等。"}
                ]
            }
        ]
    )
    desc = response.choices[0].message.content
    print(f"[VISION] {desc}")
    return desc

# ============================================================
# 3. AI Agent（支持多轮 Function Calling）
# ============================================================

class RobotDogAgent:
    def __init__(self, chat_api_key, chat_base_url, chat_model,
                 vision_api_key, vision_base_url, vision_model):
        self.chat_client = OpenAI(api_key=chat_api_key, base_url=chat_base_url)
        self.chat_model = chat_model

        self.vision_client = OpenAI(api_key=vision_api_key, base_url=vision_base_url)
        self.vision_model = vision_model

        self.system_prompt = (
            "你是一个聪明的机器狗助手，能理解中文指令。\n"
            "当用户想要移动机器狗时（包括各种模糊的说法，如‘走快一点’‘慢慢转圈’‘倒车’等），"
            "请调用 move_robot 工具，直接给出合适的线速度、角速度和持续时间。\n"
            "线速度正值为前进，负值为后退；角速度正值为左转，负值为右转。\n"
            "当用户询问‘你看到了什么’或类似问题时，请调用 describe_scene 工具获取当前画面描述，"
            "然后基于描述回答用户。\n"
            "如果只是普通聊天，请直接回复，不要调用工具。"
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
                reply = msg.content or ""
                return {"type": "chat", "reply": reply, "command_data": None}

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
                    command_info = {
                        "linear_velocity": lin,
                        "radian_velocity": ang,
                        "duration_seconds": dur
                    }
                elif tool_name == "describe_scene":
                    tool_result = describe_scene(self.vision_client, self.vision_model)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_result)
                })

            resp2 = self.chat_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                stream=False,
            )
            final_reply = resp2.choices[0].message.content or ""

            if command_info:
                return {"type": "command", "reply": final_reply, "command_data": command_info}
            else:
                return {"type": "chat", "reply": final_reply, "command_data": None}

        except Exception as e:
            return {"type": "error", "reply": f"API error: {e}", "command_data": None}

# ============================================================
# 4. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Robot Dog Control Program v5.0 (Voice + Real Robot + Vision + Qwen TTS)"
    )
    parser.add_argument("--chat_api_key", type=str,
                        default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--chat_base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--chat_model", type=str, default="deepseek-chat")
    parser.add_argument("--vision_api_key", type=str,
                        default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--vision_base_url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--vision_model", type=str, default="qwen3.6-plus")
    parser.add_argument("--voice", type=str, default="zh", choices=["zh", "en"])
    args = parser.parse_args()

    if not init_ros():
        print("FATAL: Cannot initialize ROS /cmd_vel publisher. Exiting.")
        sys.exit(1)

    init_tts_api_key()

    # 初始化语音识别器（共用视觉API key）
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
        print(f"Agent init failed: {e}")
        sys.exit(1)

    print("=" * 55)
    print("  Robot Dog Control v5.0 (Real Robot + Vision + Voice)")
    print("  " + "-" * 50)
    print("  Tools: move_robot, describe_scene")
    print("  TTS: Alibaba Cloud Qwen TTS (voice: {})".format(args.voice))
    print("  Input: 直接输入文本命令，或输入 'v'/'语音' 使用语音识别")
    print("  " + "-" * 50)
    print("  支持自然语言指令 & 图像理解（如：'你看到了什么？'）")
    print("  Exit: exit / quit / q")
    print("=" * 55)

    while True:
        try:
            text = input("\nYou (输入 v 开始语音, 或直接输入文本): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            speak_text("再见!", voice=args.voice)
            break

        if text.lower() in ["exit", "quit", "q"]:
            print("Bye!")
            speak_text("再见!", voice=args.voice)
            break

        # ---- 语音输入分支 ----
        if text.lower() in ["v", "voice", "语音"]:
            print("🎙️ 进入语音识别模式，请对准麦克风说话...")
            spoken_text = recognizer.listen(max_duration=10.0, idle_timeout=3.0)
            if not spoken_text:
                print("未识别到有效语音，返回文本输入。")
                continue
            print(f"🗣️ 语音识别结果: {spoken_text}")
            text = spoken_text  # 将识别结果作为命令文本继续处理

        if not text:
            continue

        r = agent.process(text)
        if r is None:
            print("Robot: ? Something went wrong")
            speak_text("请重试", voice=args.voice)
            continue

        if r["type"] == "command":
            print("Robot: " + r["reply"])
            cd = r["command_data"]
            print(f"      |- 线速度: {cd['linear_velocity']:.2f} m/s")
            print(f"      |- 角速度: {cd['radian_velocity']:.2f} rad/s")
            print(f"      |- 持续时间: {cd['duration_seconds']:.1f} s")
            speak_text(r["reply"], voice=args.voice)
        elif r["type"] == "chat":
            print("Robot: " + r["reply"])
            speak_text(r["reply"], voice=args.voice)
        else:
            print("Error: " + r["reply"])
            speak_text(r["reply"], voice=args.voice)

if __name__ == "__main__":
    main()