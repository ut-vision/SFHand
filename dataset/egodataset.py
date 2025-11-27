import random
import numpy as np
import pandas as pd
import torch

import decord
decord.bridge.set_bridge("native")
decord.logging.set_level(decord.logging.FATAL)

from .data_utils import video_loader
from .lmdb_utils import LMDBEngine
import ast
import clip
import torchvision.transforms as transforms


# client = Client()

class EgoHaFLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=True, tokenizer=None, crop_size=224,
                 subsample_stride=None, split='train'):
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.EgoHaFL_root = cfg.EgoHaFL_root
        self.EgoHaFL_hand_root = cfg.EgoHaFL_hand_root
        self.EgoHaFL_metadata = cfg.EgoHaFL_metadata
        self.EgoHaFL_chunk_len = cfg.EgoHaFL_video_chunk_len
        self.EgoHaFL_fps = cfg.EgoHaFL_fps
        self.clip_length = cfg.clip_length
        self.max_hand = cfg.max_hand
        self.depth_thresh = cfg.depth_threshold
        self.split = split

        self.is_trimmed = cfg.is_trimmed
        ### hardcode this for now ###
        self.narration_selection = 'random'

        if self.dataset == 'EgoHaFL':
            all_samples = pd.read_csv(self.EgoHaFL_metadata)
            self.samples = all_samples

            if cfg.EgoHaFL_metadata_aux is not None:
                self.aux_samples = pd.read_csv(cfg.EgoHaFL_metadata_aux)
                self.samples = pd.concat([self.samples, self.aux_samples])
        else:
            raise NotImplementedError
        print(len(self.samples))
        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]

        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.clip_length = cfg.clip_length
        self.clip_stride = cfg.clip_stride
        self.threads = cfg.decode_threads
        self.context_length = cfg.context_length
        print(f'sentence length {self.context_length}')
        self.multiview = cfg.multiview

        self.fast_rrc = cfg.fused_decode_crop
        self.rrc_params = (crop_size, (0.5, 1.0))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.samples)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(self.EgoHaFL_hand_root, write=False)

    def process_text(self, narration):
        ### this is a list of narrations ###
        if narration[0] == '[' and narration[-1] == ']':
            narration = ast.literal_eval(narration)
            if self.narration_selection == 'random':
                narration = random.choice(narration)
            elif self.narration_selection == 'concat':
                narration = '. '.join(narration)
            else:
                raise NotImplementedError

        return narration

    def __getitem__(self, i):
        if not hasattr(self, "_lmdb_engine"):
            self._init_lmdb_database()
        # try:
        ### get indicator ###
        curr = self.samples.iloc[i]
        curr_dataset = curr['dataset'] if 'dataset' in curr else 'EgoHaFL'
        exo_vid_path = ''
        # print(curr['video_id'],curr_dataset)
        ### get data ###

        if curr_dataset == 'EgoHaFL':

            uid, vid, start_second, end_second, narration = (
                curr['uid'], curr['video_id'], curr['start_second'], curr['end_second'], curr['caption'])
            width, height = curr['vid_w'], curr['vid_h']
            focal = curr['fx']
            # print(f'Getting ego video {vid} from {start_second} to {end_second}')
            # print('=============', self.ego4d_root, vid, self.ego4d_chunk_len)
            frames = video_loader(self.EgoHaFL_root, vid, 'mp4',
                                  start_second, end_second,
                                  chunk_len=self.EgoHaFL_chunk_len, clip_length=self.clip_length,
                                  threads=self.threads, fps=self.EgoHaFL_fps,
                                  fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)

            narration = self.process_text(narration)
            frames_rear = frames
        else:
            raise NotImplementedError

        if self.transform is not None:
            frames = frames.float() / 255.0
            frames = self.transform(frames.permute(0, 3, 1, 2))
            # frames_rear = self.transform(frames_rear.permute(0, 3, 1, 2))
            transform_no_norm = transforms.Compose(self.transform.transforms[:1])
            frames_rear = transform_no_norm(frames_rear.permute(0, 3, 1, 2))

        if self.tokenizer is not None:
            narration = narration.replace('\n', '')
            caption = self.tokenizer(narration)
        else:
            narration = narration.replace('\n', '')
            caption = clip.tokenize(narration, context_length=77, truncate=True)

        hand_pkl = self._lmdb_engine[uid]
        
        assert 16 % self.clip_length == 0, "clip length must divide 16"
        step = 16 // self.clip_length
        selected_frames = hand_pkl[::step]
        num_frames = len(selected_frames)
        
        # 预先分配数组，避免列表append和后续转换的开销
        scale_x = self.crop_size / width
        scale_y = self.crop_size / height
        scale_xy = np.array([scale_x, scale_y, scale_x, scale_y])
        
        hand_annot = {
            'camera_t': np.zeros((num_frames, self.max_hand, 3), dtype=np.float32),
            'hand_type': np.ones((num_frames, self.max_hand), dtype=np.int8) * 2,
            'global_orient': np.zeros((num_frames, self.max_hand, 1, 3, 3), dtype=np.float32),
            'hand_pose': np.zeros((num_frames, self.max_hand, 15, 3, 3), dtype=np.float32),
            'betas': np.zeros((num_frames, self.max_hand, 10), dtype=np.float32),
            'boxes': np.zeros((num_frames, self.max_hand, 4), dtype=np.float32),
            'keypoints_3d': np.zeros((num_frames, self.max_hand, 21, 3), dtype=np.float32),
            'vertices': np.zeros((num_frames, self.max_hand, 778, 3), dtype=np.float32),
            'orig_size': np.zeros((num_frames, 2), dtype=np.float32),
            'valid': np.zeros((num_frames, 1), dtype=np.int32),
        }
        
        for frame_idx, fr in enumerate(selected_frames):
            if len(fr) == 0:
                continue
            
            camera_t_z = fr['camera_t'][:, 2]
            valid_mask = (camera_t_z > 0) & (camera_t_z < self.depth_thresh)
            valid_indices = np.where(valid_mask)[0]
            h = min(len(valid_indices), self.max_hand)
            
            if h == 0:
                continue
                
            valid_indices = valid_indices[:h]
            
            hand_annot['camera_t'][frame_idx, :h] = fr['camera_t'][valid_indices]
            hand_annot['hand_type'][frame_idx, :h] = fr['is_right'][valid_indices]
            hand_annot['global_orient'][frame_idx, :h] = fr['mano_params']['global_orient'][valid_indices]
            hand_annot['hand_pose'][frame_idx, :h] = fr['mano_params']['hand_pose'][valid_indices]
            hand_annot['betas'][frame_idx, :h] = fr['mano_params']['betas'][valid_indices]
            hand_annot['keypoints_3d'][frame_idx, :h] = fr['keypoints_3d'][valid_indices]
            hand_annot['vertices'][frame_idx, :h] = fr['vertices'][valid_indices]
            hand_annot['valid'][frame_idx, 0] = h
            
            hand_annot['boxes'][frame_idx, :h, :2] = fr['box_center'][valid_indices]
            hand_annot['boxes'][frame_idx, :h, 2] = fr['box_size'][valid_indices]
            hand_annot['boxes'][frame_idx, :h, 3] = fr['box_size'][valid_indices]
            
            hand_annot['boxes'][frame_idx, :h] *= scale_xy
            hand_annot['boxes'][frame_idx, :h] /= self.crop_size
            
            hand_annot['orig_size'][frame_idx] = [width, height]

        hand_annot['camera_t'] = torch.from_numpy(hand_annot['camera_t']).to(torch.float16)
        hand_annot['hand_type'] = torch.from_numpy(hand_annot['hand_type']).to(torch.int)
        hand_annot['global_orient'] = torch.from_numpy(hand_annot['global_orient']).to(torch.float16)
        hand_annot['hand_pose'] = torch.from_numpy(hand_annot['hand_pose']).to(torch.float16)
        hand_annot['betas'] = torch.from_numpy(hand_annot['betas']).to(torch.float16)
        hand_annot['boxes'] = torch.from_numpy(hand_annot['boxes']).to(torch.float16)
        hand_annot['keypoints_3d'] = torch.from_numpy(hand_annot['keypoints_3d']).to(torch.float16)
        hand_annot['vertices'] = torch.from_numpy(hand_annot['vertices']).to(torch.float16)
        hand_annot['orig_size'] = torch.from_numpy(hand_annot['orig_size']).to(torch.float16)
        hand_annot['valid'] = torch.from_numpy(hand_annot['valid']).to(torch.int)
        hand_annot['uid'] = uid
        hand_annot['focal'] = focal
        return frames, frames_rear, caption, hand_annot

    def close(self, ):
        if hasattr(self, "_lmdb_engine"):
            self._lmdb_engine.close()
