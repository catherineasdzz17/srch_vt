import torch
import numpy as np
from pydub import AudioSegment


# def split_audio_with_overlap(file_path, segment_length_ms=30000, overlap_length_ms=5000):
#     '''
#     分割音频为小段，并带有重叠。
#     - file_path: 输入音频路径。
#     - segment_length_ms: 每段音频的长度（毫秒）。
#     - overlap_length_ms: 每段音频之间的重叠长度（毫秒）。
#     '''
#     audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)  # 加载音频
#     segments = []
#     step = segment_length_ms - overlap_length_ms  # 滑动步长
#     # 遍历整个音频
#     for i in range(0, len(audio), step):
#         segments.append(audio[i : i + segment_length_ms])
#     return segments


def split_audio_with_overlap(file_path, segment_length_ms=30000, overlap_length_ms=5000):
    '''
    分割音频为小段，并带有重叠。
    - file_path: 输入音频路径。
    - segment_length_ms: 每段音频的长度（毫秒）。
    - overlap_length_ms: 每段音频之间的重叠长度（毫秒）。
    返回音频及时间戳（秒）。
    '''
    audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1)  # 加载音频
    segments = []
    step = segment_length_ms - overlap_length_ms  # 滑动步长
    
    total_ms = len(audio)
    start_ms = 0
    # 遍历整个音频
    while start_ms < total_ms:
        end_ms = min(start_ms + segment_length_ms, total_ms)
        segment = audio[start_ms:end_ms]
        segments.append({
            "segment": segment,
            "start": start_ms / 1000.0,
            "end": end_ms / 1000.0
        })
        if end_ms ==total_ms:
            break
        start_ms += step
    return segments


def transcribe_long_audio(file_path, segment_length_sec=30, overlap_length_sec=5, model=None, processor=None):
    '''
    对长音频进行分段处理并拼接识别结果。
    返回带时间戳的识别结果列表。
    '''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    segment_length_ms = segment_length_sec * 1000
    overlap_length_ms = overlap_length_sec * 1000
    
    # 分割音频
    segments = split_audio_with_overlap(
        file_path, 
        segment_length_ms=segment_length_ms, 
        overlap_length_ms=overlap_length_ms
    )

    # 逐段识别
    results = []
    for seg_info in segments:
        seg = seg_info["segment"]
        start_s = seg_info["start"]
        end_s = seg_info["end"]

        # 音频预处理
        samples = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
        inputs = processor(samples, return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"])
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        results.append({"start": start_s, "end": end_s, "text": text})

    # 重叠去重（模糊匹配）
    max_overlap_chars = int(overlap_length_sec * 10)
    cleaned = []
    for i, cur in enumerate(results):
        if i == 0:
            cleaned.append(cur)
        else:
            prev = cleaned[-1]
            cur_text = remove_overlap_fuzzy(prev["text"], cur["text"], max_overlap_chars)
            cleaned.append({**cur, "text": cur_text})

    # 合并超短片段
    MIN_SEGMENT_LENGTH = 5
    MIN_SEGMENT_DURATION = 1.0
    
    merged = []
    for seg in cleaned:
        txt = seg["text"].strip()
        if not txt:
            continue
        
        duration = seg["end"] - seg["start"]
        
        # 只合并极短且时长也很短的片段
        if (merged and 
            len(txt) < MIN_SEGMENT_LENGTH and 
            duration < MIN_SEGMENT_DURATION):
            merged[-1]["text"] += " " + txt
            merged[-1]["end"] = seg["end"]
        else:
            merged.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": txt
            })
    
    return merged


def remove_overlap_fuzzy(prev_text, cur_text, max_overlap_chars=50):
    """
    模糊匹配并移除重叠部分
    """
    if not prev_text or not cur_text:
        return cur_text
    
    tail = prev_text[-max_overlap_chars:].strip()
    
    # 从长到短尝试匹配
    for i in range(len(tail), 3, -1):  # 最少匹配3个字符
        substring = tail[-i:]
        if cur_text.startswith(substring):
            return cur_text[i:].lstrip()
    
    return cur_text