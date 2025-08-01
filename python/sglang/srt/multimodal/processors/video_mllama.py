import av
import cv2
import math
import numpy as np
import concurrent.futures
from typing import List, Optional, Union
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
)
from sglang.srt.models.video_mllama import VideoMllamaForConditionalGeneration
from sglang.srt.utils import load_image


def validate_frame_sampling(sample_indices, frames, max_missing_frames=2, max_missing_ratio=0.1):
    """
    Validate the completeness of sampled frames.
    """
    expected_count = len(sample_indices)
    actual_count = len(frames)
    missing_count = expected_count - actual_count

    if missing_count <= 0:
        return

    missing_ratio = missing_count / expected_count

    if missing_count > max_missing_frames and missing_ratio > max_missing_ratio:
        raise ValueError(
            f"Too many frames missing: {missing_count}/{expected_count} "
            f"({missing_ratio:.1%}) frames missing, exceeding "
            f"{max_missing_ratio:.0%} threshold."
        )


def _get_video_sample_frames(video_stream, total_frames: int = 0, **kwargs) -> np.ndarray:
    """
    Core logic to compute video sample frame indices.
    """
    video_fps: float = kwargs.get("video_fps", 1.0)
    video_minlen: int = kwargs.get("video_minlen", 8)
    video_maxlen: int = kwargs.get("video_maxlen", 256)

    obtained_total_frames = int(video_stream.frames)

    duration = float(video_stream.duration * video_stream.time_base)
    frame_rate = float(video_stream.average_rate)
    calculated_total_frames = round(duration * frame_rate)
    assert video_fps <= frame_rate, f"Sampling frequency ({video_fps}) must be less than or equal to video frame rate ({frame_rate})"

    total_frames_num = [x for x in [total_frames, obtained_total_frames, calculated_total_frames] if x > 0]
    final_total_frames = min(total_frames_num) if total_frames_num else 0
    if final_total_frames == 0:
        raise AttributeError("Unable to obtain or calculate the total number of frames in the video.")

    target_total_frames = int(math.ceil(duration * video_fps - 1e-6))
    sample_frames = max(target_total_frames, video_minlen)
    sample_frames = min(sample_frames, video_maxlen, final_total_frames)

    if target_total_frames == sample_frames and video_fps > 0 and frame_rate > 0:
        sample_indices = np.arange(target_total_frames, dtype=np.int32)
        sample_indices = (sample_indices * frame_rate / video_fps).astype(np.int32)
    else:
        sample_indices = np.linspace(0, final_total_frames - 1, sample_frames).astype(np.int32)

    return sample_indices


def _get_cv2_video_sample_frames(video_path: str, total_frames: int = 0, **kwargs) -> np.ndarray:
    container = av.open(video_path, "r")
    video_stream = next(stream for stream in container.streams if stream.type == "video")
    sample_indices = _get_video_sample_frames(video_stream, total_frames=total_frames, **kwargs)
    return sample_indices


def get_video_sample_frames_av(video_path: str, **kwargs) -> List[Image.Image]:
    container = av.open(video_path, "r")
    video_stream = next(stream for stream in container.streams if stream.type == "video")

    sample_indices = _get_video_sample_frames(video_stream, **kwargs)
    sample_indices_set = set(sample_indices)

    frames: List[Image.Image] = []

    container.seek(0)
    for frame_idx, frame in enumerate(container.decode(video_stream)):
        if frame_idx in sample_indices_set:
            frames.append(frame.to_image())
        if len(frames) == len(sample_indices):
            break

    validate_frame_sampling(sample_indices, frames)
    
    return frames


def get_cv2_video_sample_frames_multithread(video_path: str, **kwargs) -> List[Image.Image]:
    num_threads: int = kwargs.get("frame_extract_num_threads", 4)
    num_threads = int(num_threads)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices = _get_cv2_video_sample_frames(video_path, total_frames=total_frames, **kwargs)

    unique_frames: List[Optional[np.ndarray]] = [None] * len(frame_indices)
    index_map = {idx: pos for pos, idx in enumerate(frame_indices)}

    chunks = np.array_split(frame_indices, min(num_threads, len(frame_indices)))

    def worker(chunk_indices):
        local_cap = cv2.VideoCapture(video_path)
        if not local_cap.isOpened():
            return

        if chunk_indices[0] > 0:
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_indices[0])

        frame_idx_cursor = chunk_indices[0]
        chunk_cursor = 0

        while chunk_cursor < len(chunk_indices):
            target_idx = chunk_indices[chunk_cursor]
            ok = local_cap.grab()
            if not ok:
                break

            if frame_idx_cursor == target_idx:
                ret, frame = local_cap.retrieve()
                if ret:
                    unique_pos = index_map[target_idx]
                    unique_frames[unique_pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chunk_cursor += 1
            frame_idx_cursor += 1
        local_cap.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(worker, [chunk for chunk in chunks if len(chunk) > 0]))

    pil_frames = [Image.fromarray(frame) for frame in unique_frames if frame is not None]

    validate_frame_sampling(frame_indices, pil_frames)

    if not pil_frames:
        return get_video_sample_frames_av(video_path, **kwargs)

    return pil_frames


class VideoMllamaImageProcessor(BaseMultimodalProcessor):
    models = [VideoMllamaForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        
        # Video mllama specific configuration
        self.image_placeholder = "<image>"
        self.video_placeholder = "<video>"
        
        # Token IDs and tokens
        if not hasattr(self._processor.tokenizer, "image_token"):
            self.image_token = "<|image|>"
            self.image_token_id = self._processor.tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token = self._processor.tokenizer.image_token
            self.image_token_id = self._processor.tokenizer.image_token_id

        if not hasattr(self._processor.tokenizer, "video_token"):
            self.video_token = "<|video|>"
            self.video_token_id = self._processor.tokenizer.convert_tokens_to_ids(self.video_token)
        else:
            self.video_token = self._processor.tokenizer.video_token
            self.video_token_id = self._processor.tokenizer.video_token_id

        # Set required token IDs for base class
        self.IM_TOKEN_ID = self.image_token_id
        self.VIDEO_TOKEN_ID = self.video_token_id

        # Video sampling configuration
        self.video_fps = getattr(_processor.image_processor, "video_fps", 1.0)
        self.video_minlen = getattr(_processor.image_processor, "video_minlen", 8)
        self.video_maxlen = getattr(_processor.image_processor, "video_maxlen", 256)
        self.frame_extract_num_threads = getattr(_processor.image_processor, "frame_extract_num_threads", 4)
        self.extract_frame_func = getattr(_processor.image_processor, "extract_frame_func", "cv2")  # Options: "av", "cv2"
        
        # Max image tiles configuration
        self.max_image_tiles = 1

    def _convert_video_tokens_to_image_tokens(self, input_ids, frame_num_per_video):
        """
        Convert video tokens to image tokens
        """
        input_ids_np = np.array(input_ids, dtype=np.int64)
        if self.VIDEO_TOKEN_ID in input_ids_np:
            total_vid_num = np.sum(input_ids_np == self.VIDEO_TOKEN_ID)
            f_num_per_vid = frame_num_per_video[:total_vid_num]
            convert_input_ids_list = []
            vid_idx = 0
            for token_id in input_ids_np:
                if token_id == self.VIDEO_TOKEN_ID:
                    vid_len = f_num_per_vid[vid_idx]
                    vid_idx += 1
                    convert_input_ids_list.extend([self.IM_TOKEN_ID] * vid_len)
                else:
                    convert_input_ids_list.append(token_id)
            convert_input_ids = np.array(convert_input_ids_list, dtype=np.int64)
        else:
            convert_input_ids = input_ids_np
        return convert_input_ids

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if isinstance(input_text, list):
            assert len(input_text) and isinstance(input_text[0], int)
            input_text = self._processor.tokenizer.decode(input_text)

        # Replace user-facing placeholders with tokenizer's special tokens
        processed_text = input_text.replace(self.image_placeholder, self.image_token)
        processed_text = processed_text.replace(self.video_placeholder, self.video_token)

        # Validate media token counts
        n_images_in_text = processed_text.count(self.image_token)
        n_videos_in_text = processed_text.count(self.video_token)
        
        n_images_provided = len(image_data) if image_data else 0
        n_videos_provided = len(request_obj.video_data) if hasattr(request_obj, 'video_data') and request_obj.video_data else 0
        
        if n_images_in_text != n_images_provided:
            raise ValueError(
                f"Number of image tokens ({n_images_in_text}) does not match number of images provided ({n_images_provided})."
            )
        
        if n_videos_in_text != n_videos_provided:
            raise ValueError(
                f"Number of video tokens ({n_videos_in_text}) does not match number of videos provided ({n_videos_provided})."
            )

        # Directly tokenize the text
        input_ids = self._processor.tokenizer(
            processed_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.flatten()

        # Collect all visual data
        all_images = []
        all_frames = []
        frame_num_per_video = []

        # Process images directly
        if image_data:
            for image in image_data:
                if isinstance(image, str) or isinstance(image, bytes):
                    img, _ = load_image(image)
                    all_images.append(img)
                else:
                    all_images.append(image)

        # Process videos - extract frames
        if hasattr(request_obj, 'video_data') and request_obj.video_data:
            for video_path in request_obj.video_data:
                if isinstance(video_path, str):
                    # Extract frames from video
                    sampling_kwargs = {
                        "video_fps": self.video_fps,
                        "video_minlen": self.video_minlen,
                        "video_maxlen": self.video_maxlen,
                        "frame_extract_num_threads": self.frame_extract_num_threads,
                    }
                    try:
                        if self.extract_frame_func == "cv2":
                            frames = get_cv2_video_sample_frames_multithread(video_path, **sampling_kwargs)
                        else:  # "av"
                            frames = get_video_sample_frames_av(video_path, **sampling_kwargs)
                        
                        # Add frames to all_frames
                        all_frames.extend(frames)
                        frame_num_per_video.append(len(frames))
                        
                    except Exception as e:
                        raise ValueError(f"Video processing failed: {e}")

        input_ids = self._convert_video_tokens_to_image_tokens(input_ids, frame_num_per_video)
        mm_items = []
        image_features = self._processor.image_processor(
                images=all_images,
                return_tensors="pt",
                max_image_tiles=self.max_image_tiles
            )
        video_features = self._processor.image_processor(
                images=all_frames,
                return_tensors="pt",
                max_image_tiles=self.max_image_tiles
            )
        mm_items.append(
            MultimodalDataItem(
                pixel_values=image_features["pixel_values"],
                aspect_ratio_id=image_features["aspect_ratio_ids"],
                aspect_ratio_mask=image_features["aspect_ratio_mask"],
                modality=Modality.IMAGE,
            )
        )
        mm_items.append(
            MultimodalDataItem(
                pixel_values=video_features["pixel_values"],
                aspect_ratio_id=video_features["aspect_ratio_ids"],
                aspect_ratio_mask=video_features["aspect_ratio_mask"],
                modality=Modality.VIDEO,
            )
        )
        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "frame_num_per_video": frame_num_per_video,
        } 