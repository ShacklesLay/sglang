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


    def _get_cross_attention_token_mask(self, input_ids, frame_num_per_video, cross_attention_token_mask_pad_token_id=-100):
        """
        Generate a cross-attention-token-mask for each input_tokens in the input sequence.
        This function implements a causal attention logic:
        - A text token can see all image tokens that appeared before it.
        - An image token can see itself and all image tokens that appeared before it.
        """
        # 1. Convert video tokens to image tokens
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

        # 2. Generate the sparse attention mask based on causal visibility
        is_image = convert_input_ids == self.IM_TOKEN_ID
        # Cumulative count of images up to and including the current position
        image_count_cumulative = np.cumsum(is_image)
        # Cumulative count of images up to the previous position
        image_count_before = np.pad(image_count_cumulative[:-1], (1, 0), "constant", constant_values=0)

        # For text tokens, num_seen = image_count_before.
        # For image tokens, num_seen = image_count_cumulative (sees itself).
        num_images_seen = np.where(is_image, image_count_cumulative, image_count_before)

        # Convert num_seen to sparse mask value (num_seen - 1).
        vision_masks = np.full(len(convert_input_ids), cross_attention_token_mask_pad_token_id, dtype=np.int64)
        valid_mask = num_images_seen > 0
        vision_masks[valid_mask] = num_images_seen[valid_mask] - 1

        return vision_masks.tolist(), convert_input_ids.tolist()

    def _compute_vision_position_ids(self, input_ids):
        """
        Compute vision position ids for image tokens in the sequence.
        This follows the VideoMllama implementation.
        
        Note: vision_position_ids only tracks the position of <image> tokens in the input sequence,
        not individual vision patches. It has shape (1, num_image_tokens) and is not affected by pooling.
        """
        # Find positions of image tokens
        input_ids_arr = np.array(input_ids, dtype=np.int64)
        image_mask = input_ids_arr == self.IM_TOKEN_ID
        
        # Get positions where image tokens appear
        if np.any(image_mask):
            # Compute cumulative position ids
            cumulative_positions = np.cumsum(np.ones_like(input_ids_arr), dtype=np.int64) - 1
            # Extract positions of image tokens
            vision_position_ids = cumulative_positions[image_mask]
            return vision_position_ids.tolist()
        else:
            return []

    def _convert_sparse_cross_attention_mask_to_dense(self, cross_attention_token_masks, n_tiles, max_tiles_per_image, cross_attention_token_mask_pad_token_id=-100):
        """
        Convert the cross attention mask indices to a cross attention mask 4D array.
        """
        seq_len = len(cross_attention_token_masks)
        max_num_images = len(n_tiles) if n_tiles else 0

        cross_attention_mask = np.zeros(
            shape=(1, seq_len, max_num_images, max_tiles_per_image),
            dtype=np.int64,
        )

        if max_num_images == 0:
            return cross_attention_mask

        sparse_mask = np.array(cross_attention_token_masks)
        # For each image, find all text tokens that are allowed to see it.
        # A token with sparse_mask value N can see all images with index i <= N.
        for image_idx, mask_n_tiles in enumerate(n_tiles):
            # Find all token positions where the sparse mask value is >= the current image's index.
            # This correctly implements the causal logic.
            visible_token_indices = (sparse_mask >= image_idx) & (sparse_mask != cross_attention_token_mask_pad_token_id)
            # Set the attention mask to 1 for these tokens and the current image.
            cross_attention_mask[0, visible_token_indices, image_idx, :mask_n_tiles] = 1

        return cross_attention_mask

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
        
        n_images_provided = len(image_data) if image_data and image_data[0] else 0
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
        if image_data and image_data[0]:
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

        # Convert video tokens to image tokens and get cross attention mask
        cross_attention_token_mask, converted_input_ids = self._get_cross_attention_token_mask(
            input_ids.tolist(), frame_num_per_video
        )
        
        # Compute vision position ids
        vision_position_ids = self._compute_vision_position_ids(converted_input_ids)
        
        mm_items = []
        cross_attention_mask = None
        
        # Process images and videos together
        all_media = all_images + all_frames
        if all_media:
            features = self._processor.image_processor(
                images=all_media,
                return_tensors="pt",
                max_image_tiles=self.max_image_tiles
            )
            
            # Split features back to images and videos
            n_images = len(all_images)
            n_frames = len(all_frames)
            
            if n_images > 0:
                image_features = {
                    "pixel_values": features["pixel_values"][:,:n_images,],
                    "aspect_ratio_ids": features["aspect_ratio_ids"][:,:n_images,],
                    "aspect_ratio_mask": features["aspect_ratio_mask"][:,:n_images,],
                }
                mm_items.append(
                    MultimodalDataItem(
                        pixel_values=image_features["pixel_values"],
                        aspect_ratio_id=image_features["aspect_ratio_ids"],
                        aspect_ratio_mask=image_features["aspect_ratio_mask"],
                        modality=Modality.IMAGE,
                    )
                )
            
            if n_frames > 0:
                video_features = {
                    "pixel_values": features["pixel_values"][:,n_images:,],
                    "aspect_ratio_ids": features["aspect_ratio_ids"][:,n_images:,],
                    "aspect_ratio_mask": features["aspect_ratio_mask"][:,n_images:,],
                }
                mm_items.append(
                    MultimodalDataItem(
                        pixel_values=video_features["pixel_values"],
                        aspect_ratio_id=video_features["aspect_ratio_ids"],
                        aspect_ratio_mask=video_features["aspect_ratio_mask"],
                        modality=Modality.VIDEO,
                    )
                )
            
            # Compute dense cross attention mask
            if "num_tiles" in features:
                n_tiles = features["num_tiles"][0] # Processor process one req at a time
                cross_attention_mask = self._convert_sparse_cross_attention_mask_to_dense(
                    cross_attention_token_mask, n_tiles, self.max_image_tiles
                )
                # Convert to tensor and apply mask logic
                import torch
                cross_attention_mask = torch.from_numpy(cross_attention_mask)
                
                # Get vision config for num_vision_tokens calculation
                try:
                    vision_config = getattr(self.hf_config, 'vision_config', None)
                    if vision_config is None:
                        # Fallback values
                        image_size = 560
                        patch_size = 14
                        merge_size = getattr(self._processor.image_processor, 'merge_size', None)
                        merge_mode = getattr(self._processor.image_processor, 'merge_mode', None)
                    else:
                        image_size = getattr(vision_config, 'image_size', 560)
                        patch_size = getattr(vision_config, 'patch_size', 14)
                        merge_size = getattr(vision_config, 'merge_size', None)
                        merge_mode = getattr(vision_config, 'merge_mode', None)
                    
                    num_vision_tokens = (image_size // patch_size) ** 2 + 1
                    
                    # Apply merge/pooling if configured
                    if merge_mode is not None and merge_size is not None:
                        height = width = image_size // patch_size
                        import math
                        if merge_mode in ["average", "max", "channelFusion"]:
                            new_height = height // merge_size
                            new_width = width // merge_size
                        elif merge_mode == "bilinear":
                            new_height = math.ceil(height / merge_size)
                            new_width = math.ceil(width / merge_size)
                        else:
                            new_height = height
                            new_width = width
                        num_vision_tokens = new_height * new_width + 1
                    
                    # Expand cross attention mask to include all vision tokens
                    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
                    # Reshape to (1, seq_len, total_vision_tokens)
                    cross_attention_mask = cross_attention_mask.view(1, len(converted_input_ids), -1)
                    
                    # Convert to boolean mask (True where we should attend)
                    cross_attention_mask = cross_attention_mask > 0
                    # Flatten the mask
                    cross_attention_mask = cross_attention_mask.view(-1)
                    
                except Exception as e:
                    print(f"Warning: Failed to compute detailed cross attention mask: {e}")
                    cross_attention_mask = None

        result = {
            "input_ids": converted_input_ids,
            "mm_items": mm_items,
            "frame_num_per_video": frame_num_per_video,
        }
        
        # Add cross attention mask and vision position ids if computed
        if cross_attention_mask is not None:
            result["cross_attention_mask"] = cross_attention_mask
        
        if vision_position_ids:
            import torch
            result["vision_position_ids"] = torch.tensor(vision_position_ids, dtype=torch.int64)
            
        return result 