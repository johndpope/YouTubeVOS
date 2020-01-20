################################################################################

""" Author: Deep Jigar Kotecha """

################################################################################

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib; matplotlib.use('tkagg')
import torchvision.transforms as transforms

################################################################################

from PIL import Image
from clstm import ConvLSTMCell
from dataloader import youtubeVOS
from matplotlib import pyplot as plt
from models import initializer, encoder, decoder, mean_encoder
from torch.utils.data import DataLoader, Dataset

################################################################################

lr = 1e-3
num_epochs = 10
num_workers = 4
batch_size = 4
shuffle_train = True
dataset_percent = 1

################################################################################

results = "/home/cap6412.student2/experiments/results/basemodel_"
outputdir = "/home/cap6412.student2/experiments/output/"

try:
	os.mkdir("results")
	os.mkdir("output")
except:
	raise ValueError("results dir not created!")
################################################################################

device = torch.device('cuda:0')

################################################################################

print("Loading youtubeVOS dataset ...\n")

dataset = youtubeVOS(percent = dataset_percent)
data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle_train, num_workers = num_workers)

################################################################################

loss_list = []
loss_values = []
avg_loss_values = []
total_step = len(data_loader)

################################################################################

print("Pushing the model to GPU ...\n")

init = initializer().to(device)
encoder = encoder().to(device)
mean_encoder = mean_encoder().to(device)
decoder = decoder().to(device)
clstm = ConvLSTMCell(input_size = (8, 14), input_dim = 512, hidden_dim = 512, kernel_size = (3, 3), bias = True).to(device)
reduction = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding = 1).to(device)


criterion = nn.BCELoss()
params = list(init.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(clstm.parameters()) + list(reduction.parameters()) 
optimizer = torch.optim.Adam(params, lr = lr)


################################################################################
# Training loop
print("Training started, woohoo ...\n")
error_counter = 0
for epoch in range(num_epochs):
	avg_loss = 0.0
	for i, (video, masks, video_id, frame_id) in enumerate(data_loader):	
		# print(i, "*"*10, "Video:", video.size(), "*"*10, "Mask:", masks.size())
		optimizer.zero_grad()

		video = video.to(device)
		masks = masks.to(device)
		###############################################################################
		# print("videoid: ", video_id[0])
		# print("Video: ",video.size())
		# mean_images = torch.sum(video, dim = 1)
		# print("Mean Video: ", mean_images.size())
		# npimg = mean_images[0].cpu().detach().numpy()
		# plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
		# plt.show()
		
		###############################################################################
		
		mean_images = torch.sum(video, dim = 1).to(device)

		###############################################################################


		frame_id = np.transpose(np.array(frame_id))

		x0 = video[:,0,:,:,:]
		y0 = masks[:,0,:,:,:]
		z0 = torch.cat((x0, y0), 1)
		h_x, c_x = init(z0)

		final_loss = 0.0
		y_hat_list = [y0.squeeze(1).cpu().detach().numpy()]
		y_list = []
		
		mean_encoded_images = mean_encoder(mean_images)
		# print(mean_encoded_images.size())
		# assert video size = masks size
		for x in range(1, video.size(1)):

			x_x = video[:,x,:,:,:]
			y_x = masks[:,x,:,:,:]
			y_list.append(y_x)
			x_x_hat = encoder(x_x)
			#####################################################
			new_x_x_hat = torch.cat((x_x_hat, mean_encoded_images), 1)
			new_x_x_hat = reduction(new_x_x_hat)
			h_x, c_x = clstm(new_x_x_hat, (h_x, c_x))
			#####################################################

			y_x_hat = decoder(h_x)
			loss = criterion(y_x_hat, y_x)
			y_hat_list.append(y_x_hat.squeeze(1).cpu().detach().numpy())
			final_loss += loss

		avg_loss += final_loss.item()
		final_loss.backward()
		loss_list.append(final_loss.item())
		optimizer.step()

		print ("Epoch [{}/{}] \t\t Step [{}/{}] \t\t Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, final_loss.item()))

		########################################################################################
		for j in range(y_hat_list[0].shape[0]):
			vid_name = video_id[j]
			try:
				os.mkdir(outputdir + vid_name)
			except:
				error_counter += 1

			for k in range(len(y_hat_list)):
				generated_mask = y_hat_list[k][j]
				
				# generated_mask = y_x_hat[x-1].squeeze(1).cpu().detach().numpy()
				# print(generated_mask.shape)

				PIL_image = Image.fromarray(generated_mask)
				resized_image = np.array(PIL_image.resize((1280, 720), Image.ANTIALIAS))

				resized_image[resized_image <= 0.5] = 0.0
				resized_image[resized_image > 0.5] = 1.0

				# print(len(frame_id))
				# get the base name and not the file extension which is .jpg in images but .png in submission case
				mask_name, _ = os.path.splitext(frame_id[j][k])

				path = os.path.join(outputdir, vid_name, mask_name + ".png")
				plt.imsave(path, resized_image)

		########################################################################################

		# if((i+1) % 10 == 0):
		# for i, image in enumerate(y_hat_list):
		# 	plt.imshow(image[0].squeeze(0).cpu().detach().numpy())
		# 	plt.show()
		# 	temp = y_list[i][0].squeeze(0).cpu().detach().numpy()
		# 	temp[temp >= 0.5] = 1.0 
		# 	temp[temp < 0.5] = 0.0 
		# 	plt.imshow(temp)
		# 	plt.show()

	avg_loss = avg_loss / total_step
	print("After epoch", epoch+1, "loss is", avg_loss)
	avg_loss_values.append(avg_loss)
	
	# saving the model at every 20th epoch
	# if((epoch+1) % 20 == 0):
	torch.save({
	"init": init.state_dict(),
	"encoder": encoder.state_dict(),
	"mean_encoder": mean_encoder.state_dict(),
	"decoder": decoder.state_dict(),
	"clstm": clstm.state_dict(),
	"reduction": reduction.state_dict()
	}, results + str(epoch) + ".pt")

print("Training fininshed, congratulations ...\nPlotting loss graph\n")

matplotlib.use('pdf')
plt.plot(avg_loss_values)
# plt.show()
plt.savefig(results + "loss.png")