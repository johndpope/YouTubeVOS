#############################################################################

import torch
import torch.nn as nn
import torchvision.models as models

#############################################################################



#############################################################################

class initializer(nn.Module):
	def __init__(self):
		super(initializer, self).__init__()
		
		self.vgg = models.vgg16(pretrained=True).features
		
		self.c_zero = nn.Conv2d(512, 512, kernel_size=1, stride=1)
		nn.init.xavier_uniform_(self.c_zero.weight)
		
		self.h_zero = nn.Conv2d(512, 512, kernel_size=1, stride=1)
		nn.init.xavier_uniform_(self.h_zero.weight)

		self.four_to_three = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding = 1)
		self.relu = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(3) 
		self.bn2 = nn.BatchNorm2d(512) 
		self.bn3 = nn.BatchNorm2d(512) 
		# self.four_to_three = nn.Sequential(
		# 	nn.Conv2d(4, 512, kernel_size=3, stride=1, padding = 1),
		# 	nn.BatchNorm2d(512),
		# 	nn.ReLU(),
		# 	nn.Conv2d(512, 256, kernel_size=3, stride=1, padding = 1),
		# 	nn.BatchNorm2d(256),
		# 	nn.ReLU(),
		# 	nn.Conv2d(256, 128, kernel_size=3, stride=1, padding = 1),
		# 	nn.BatchNorm2d(128),
		# 	nn.ReLU(),
		# 	nn.Conv2d(128, 64, kernel_size=3, stride=1, padding = 1),
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU(),
		# 	nn.Conv2d(64, 32, kernel_size=3, stride=1, padding = 1),
		# 	nn.BatchNorm2d(32),
		# 	nn.ReLU(),
		# 	nn.Conv2d(32, 16, kernel_size=3, stride=1, padding = 1),
		# 	nn.BatchNorm2d(16),
		# 	nn.ReLU(),
		# 	nn.Conv2d(16, 3, kernel_size=3, stride=1, padding = 1),
		# 	)
		# nn.init.xavier_uniform_(self.four_to_three.weight)


	def forward(self, x):
		x = self.relu(self.bn1(self.four_to_three(x)))
		# x = self.four_to_three(x)
		x = self.vgg(x)
		# y = self.h_zero(x)
		# z = self.c_zero(x)
		y = self.relu(self.bn2(self.h_zero(x)))
		z = self.relu(self.bn3(self.c_zero(x)))
		return y, z

################################################################################

class encoder(nn.Module):
	def __init__(self):
		super(encoder, self).__init__()
		self.vgg = models.vgg16(pretrained=True).features
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm2d(512) 
		self.conv = nn.Conv2d(512, 512, kernel_size=1, stride=1)
		nn.init.xavier_uniform_(self.conv.weight)


	def forward(self, x):
		x = self.vgg(x)
		x = self.relu(self.bn(self.conv(x)))
		# x = self.conv(x)
		return x

################################################################################

class decoder(nn.Module):
	def __init__(self):
		super(decoder, self).__init__()
		self.network = nn.Sequential(
			############################################################################
			nn.ConvTranspose2d(512, 512, 5, stride = 2, padding = 2, output_padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			############################################################################
			nn.ConvTranspose2d(512, 256, 5, stride = 2, padding = 2, output_padding = 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			############################################################################
			nn.ConvTranspose2d(256, 128, 5, stride = 2, padding = 2, output_padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			############################################################################
			nn.ConvTranspose2d(128, 64, 5, stride = 2, padding = 2, output_padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			############################################################################
			nn.ConvTranspose2d(64, 64, 5, stride = 2, padding = 2, output_padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			############################################################################
			nn.Conv2d(64, 1, 3, padding = 1),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		for layer in self.network[:-1]:
			if 'ConvTranspose' in layer.__class__.__name__:
				nn.init.xavier_uniform_(layer.weight)
			# elif 'BatchNorm' in layer.__class__.__name__:
			#     layer.weight.data.normal_(1.0, 0.02)
			#     layer.bias.data.fill_(0)
			# else:
			#     pass

	def forward(self, x):
		return self.network(x)
#############################################################################

class mean_encoder(nn.Module):
	def __init__(self):
		super(mean_encoder, self).__init__()
		self.vgg = models.vgg16(pretrained=True).features
		# self.relu = nn.ReLU()
		# self.bn = nn.BatchNorm2d(512) 
		# self.conv = nn.Conv2d(512, 512, kernel_size=1, stride=1)
		# nn.init.xavier_uniform_(self.conv.weight)


	def forward(self, x):
		x = self.vgg(x)
		# x = self.relu(self.bn(self.conv(x)))
		# x = self.conv(x)
		return x

################################################################################
# summaries of networks

# from torchsummary import summary
# device = torch.device('cuda:0')

################################################################################

# init = initializer().to(device)

# x0 = torch.ones(2, 3, 256, 448).to(device) # video
# y0 = torch.ones(2, 1, 256, 448).to(device) # mask
# z0 = torch.cat((x0, y0), 1)
# print(z0.size())

# summary(init, z0)
# c0, h0 = init(z0)
# print(c0.size())
# print(h0.size())

################################################################################

# encoder = encoder().to(device)
# x1 = encoder(torch.ones(1, 3, 256, 448).to(device))
# summary(encoder, x1)
# print(x0.size())

################################################################################
# from clstm import ConvLSTMCell

# clstm = ConvLSTMCell(input_size = (8, 14), input_dim = 512, hidden_dim = 512, kernel_size = (3, 3), bias = False).to(device)

# h0 = torch.randn((2, 512, 8, 14)).to(device)
# c0 = torch.randn((2, 512, 8, 14)).to(device)
# x0 = torch.randn((2, 512, 8, 14)).to(device)

# from torchsummaryX import summary
# summary(clstm, x1, (h0, c0))
# h1, c1 = clstm(x1, (h0, c0))
# print(h1.size())
# print(c1.size())

################################################################################

# from torchsummary import summary
# decoder = decoder().to(device)
# summary(decoder, (512, 8, 14))
# yt = decoder(h1)
# print(yt.size())

################################################################################