import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class youtubeVOS(Dataset):
	def __init__(self, dir = "/home/course.cap6412/youtubeVOS/", percent = 100, all_frames = False, train = True):
		super(youtubeVOS, self).__init__()
		self.dir = dir
		self.video_transformations = transforms.Compose([transforms.Resize((256, 448)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		self.mask_transformations = transforms.Compose([transforms.Resize((256, 448)), transforms.ToTensor()])

		self.train = train
		self.all_frames = all_frames
		
		self.train_all_images = 'train_all_frames/JPEGImages/'
		self.valid_all_images = 'valid_all_frames/JPEGImages/'

		self.train_images = 'train/JPEGImages/'
		self.valid_images = 'valid/JPEGImages/'

		self.train_annotations = 'train/Annotations/'
		self.valid_annotations = 'valid/Annotations/'

		self.abs_dir_images = ""
		self.abs_dir_masks = ""

		if(self.all_frames):
			if(self.train):
				self.abs_dir_images = os.path.join(self.dir, self.train_all_images)
				self.abs_dir_masks = os.path.join(self.dir, self.train_annotations)

			else:
				self.abs_dir_images = os.path.join(self.dir, self.valid_all_images)
				self.abs_dir_masks = os.path.join(self.dir, self.valid_annotations)
		else:
			if(self.train):
				self.abs_dir_images = os.path.join(self.dir, self.train_images)
				self.abs_dir_masks = os.path.join(self.dir, self.train_annotations)
			else:
				self.abs_dir_images = os.path.join(self.dir, self.valid_images)
				self.abs_dir_masks = os.path.join(self.dir, self.valid_annotations)


		self.video_names = os.listdir(self.abs_dir_images)
		self.mask_video_names = os.listdir(self.abs_dir_masks)

		self.limit = int((percent/100) * len(self.video_names))

	def __getitem__(self, index):
		if index >= self.limit:
			raise ValueError('Error loading data!')
		else:
			abs_video_name = os.path.join(self.abs_dir_images, self.video_names[index])
			abs_file_name = os.path.join(self.abs_dir_masks, self.mask_video_names[index])

			self.frames = sorted(os.listdir(abs_video_name))
			self.mask_frames = sorted(os.listdir(abs_file_name))

			# print(self.frames,"\n\n")

			self.frame_list = []
			self.mask_frame_list = []

			true_len = len(self.frames)
			for i, frame in enumerate(self.frames):
				if(i>=5):
					break
				abs_name = os.path.join(abs_video_name, frame)
				frame = Image.open(abs_name)
				# if true_len < 5:
				# 	plt.imshow(frame)
				# 	plt.show()
				frame = self.video_transformations(frame)
				self.frame_list.append(frame)

			while(len(self.frame_list) < 5):
				self.frame_list.append(self.frame_list[-1])
				self.frames.append(self.frames[-1])

			true_len_masks = len(self.mask_frames)
			for i, mask_frame in enumerate(self.mask_frames):
				if(i>=5):
					break
				abs_mask_name = os.path.join(abs_file_name, mask_frame)
				mask_frame = Image.open(abs_mask_name)
				# if true_len_masks < 5:
				# 	plt.imshow(frame)
				# 	plt.show()
				mask_frame = self.mask_transformations(mask_frame)
				maximum = torch.max(mask_frame)
				if maximum>0:
					mask_frame = mask_frame / maximum 
				mask_frame[mask_frame != 1.0] = 0.0
				# print(torch.unique(mask_frame))
				self.mask_frame_list.append(mask_frame)
			
			while(len(self.mask_frame_list) < 5):
				self.mask_frame_list.append(self.mask_frame_list[-1])

			self.video = torch.stack(self.frame_list)
			self.masked_video = torch.stack(self.mask_frame_list)

		return self.video, self.masked_video, self.video_names[index], self.frames

	def __len__(self):
		return self.limit