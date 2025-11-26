import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch
import os

import decord
from decord import cpu
decord.bridge.set_bridge("native")
decord.logging.set_level(decord.logging.FATAL)

import io
from ipdb import set_trace
from .data_utils import video_loader
from .lmdb_utils import LMDBEngine
# from petrel_client.client import Client
import ast
import clip
import pickle
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

            # # Only read the fisrt half frames
            # frames = video_loader(self.ego4d_root, vid, 'mp4',
            #                       start_second, start_second + (end_second - start_second) / 2,
            #                       chunk_len=self.ego4d_chunk_len, clip_length=self.clip_length // 2,
            #                       threads=self.threads, fps=self.ego4d_fps,
            #                       fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)

            narration = self.process_text(narration)
            frames_rear = frames
            exo_frames = torch.zeros_like(frames)
        else:
            raise NotImplementedError

        raw_caption = narration

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

        # hand_pkl = pickle.load(open(os.path.join(self.ego4d_hand_root, f'{uid}.pkl'), 'rb'))
        hand_pkl = self._lmdb_engine[uid]
        hand_annot = {'camera_t': [], 'hand_type': [], 'global_orient': [], 'hand_pose': [], 'betas': [],
                      'boxes': [], 'keypoints_3d': [], 'valid': [], 'vertices': [], 'orig_size':[], 'focal': []}

        # Only load the second half hand annotations
        assert 16 % self.clip_length == 0, "clip length must divide 16"
        step = 16 // self.clip_length
        for fr in hand_pkl[::step]:
            hand_annot['camera_t'].append(np.zeros((self.max_hand, 3)))
            hand_annot['hand_type'].append(np.ones((self.max_hand,), dtype=np.int8) * 2)
            hand_annot['global_orient'].append(np.zeros((self.max_hand, 1, 3, 3)))
            hand_annot['hand_pose'].append(np.zeros((self.max_hand, 15, 3, 3)))
            hand_annot['betas'].append(np.zeros((self.max_hand, 10)))
            hand_annot['boxes'].append(np.zeros((self.max_hand, 4)))
            hand_annot['keypoints_3d'].append(np.zeros((self.max_hand, 21, 3)))
            hand_annot['vertices'].append(np.zeros((self.max_hand, 778, 3)))
            hand_annot['orig_size'].append(np.zeros((2,)))
            hand_annot['valid'].append(np.zeros((1,)))

            if len(fr) == 0:
                continue
            # Complete the just-added annotation
            valid_indices = np.where((fr['camera_t'][:, 2] > 0) & (fr['camera_t'][:, 2] < self.depth_thresh))[0]
            h = min(len(valid_indices), self.max_hand)
            valid_indices = valid_indices[:h]
            hand_annot['camera_t'][-1][:h] = fr['camera_t'][valid_indices]
            hand_annot['hand_type'][-1][:h] = fr['is_right'][valid_indices]
            hand_annot['global_orient'][-1][:h] = fr['mano_params']['global_orient'][valid_indices]
            hand_annot['hand_pose'][-1][:h] = fr['mano_params']['hand_pose'][valid_indices]
            hand_annot['betas'][-1][:h] = fr['mano_params']['betas'][valid_indices]
            hand_annot['keypoints_3d'][-1][:h] = fr['keypoints_3d'][valid_indices]
            hand_annot['vertices'][-1][:h] = fr['vertices'][valid_indices]
            hand_annot['valid'][-1] += h  # Number of valid hands

            hand_annot['boxes'][-1][:h, :2] = fr['box_center'][valid_indices]
            hand_annot['boxes'][-1][:h, 2] = fr['box_size'][valid_indices]
            hand_annot['boxes'][-1][:h, 3] = fr['box_size'][valid_indices]

            scale_x = self.crop_size / width
            scale_y = self.crop_size / height

            hand_annot['boxes'][-1][:h, 0] *= scale_x  # cx
            hand_annot['boxes'][-1][:h, 1] *= scale_y  # cy
            hand_annot['boxes'][-1][:h, 2] *= scale_x  # w
            hand_annot['boxes'][-1][:h, 3] *= scale_y  # h

            # 如果后续需要归一化到 [0, 1]
            hand_annot['boxes'][-1][:h, :] /= self.crop_size

            hand_annot['orig_size'][-1][:] = width, height

        for k, v in hand_annot.items():
            if k in ["hand_type", "valid"]:
                hand_annot[k] = torch.from_numpy(np.array(v)).to(torch.int)
            else:
                hand_annot[k] = torch.from_numpy(np.array(v)).to(torch.float16)
        hand_annot['uid'] = uid
        hand_annot['focal'] = focal
        return frames, frames_rear, caption, hand_annot

    def close(self, ):
        if hasattr(self, "_lmdb_engine"):
            self._lmdb_engine.close()
