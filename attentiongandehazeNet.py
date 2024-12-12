# @author: hayat
import torch
import torch.nn as nn
import math

class attentionDehaze_Net(nn.Module):

	def __init__(self):
		super(attentionDehaze_Net, self).__init__()
		
		# LightDehazeNet Architecture 
		self.relu = nn.ReLU(inplace=True)

		self.e_conv_layer1 = nn.Conv2d(3,8,1,1,0,bias=True) 
		self.e_conv_layer2 = nn.Conv2d(8,8,3,1,1,bias=True) 
		self.e_conv_layer3 = nn.Conv2d(8,8,5,1,2,bias=True) 
		self.e_conv_layer4 = nn.Conv2d(16,16,7,1,3,bias=True) 
		self.e_conv_layer5 = nn.Conv2d(16,16,3,1,1,bias=True) 
		self.e_conv_layer6 = nn.Conv2d(16,16,3,1,1,bias=True) 
		self.e_conv_layer7 = nn.Conv2d(32,32,3,1,1,bias=True)
		self.e_conv_layer8 = nn.Conv2d(56,3,3,1,1,bias=True)

		self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
		self.max_pool = nn.AdaptiveMaxPool2d((1,1))
		self.mlp = nn.Sequential(
            			nn.Conv2d(56, 8,kernel_size=1,bias=False),
            			nn.ReLU(),
            			nn.Conv2d(8, 56,kernel_size=1,bias=False)
		 )


		self.sigmoid=nn.Sigmoid()
	def forward(self, img):
		pipeline = []
		pipeline.append(img)

		conv_layer1 = self.relu(self.e_conv_layer1(img))
		conv_layer2 = self.relu(self.e_conv_layer2(conv_layer1))
		conv_layer3 = self.relu(self.e_conv_layer3(conv_layer2))

		# concatenating conv1 and conv3
		concat_layer1 = torch.cat((conv_layer1,conv_layer3), 1)
		

        
		conv_layer4 = self.relu(self.e_conv_layer4(concat_layer1))
		conv_layer5 = self.relu(self.e_conv_layer5(conv_layer4))
		conv_layer6 = self.relu(self.e_conv_layer6(conv_layer5))

		# concatenating conv4 and conv6
		concat_layer2 = torch.cat((conv_layer4, conv_layer6), 1)
		
		conv_layer7= self.relu(self.e_conv_layer7(concat_layer2))

		# concatenating conv2, conv5, and conv7
		concat_layer3 = torch.cat((conv_layer2,conv_layer5,conv_layer7),1)

        
		avg_feat_channel = self.mlp(self.avg_pool(concat_layer3))
		max_feat_channel = self.mlp(self.max_pool(concat_layer3))
		att_feat_channel = avg_feat_channel + max_feat_channel
		att_weight_channel = self.sigmoid(att_feat_channel)
		concat_layer3= concat_layer3*att_weight_channel

		#print(concat_layer3.shape)

		#print(concat_layer3.shape)

		conv_layer8 = self.relu(self.e_conv_layer8(concat_layer3))


		dehaze_image = self.relu((conv_layer8 * img) - conv_layer8 + 1) 
		#J(x) = clean_image, k(x) = x8, I(x) = x, b = 1
		
		
		return dehaze_image 

		


			

			
			
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__() 
        self.Conv = nn.Sequential(
        nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(1)
        )
        self.sigmoid=nn.Sigmoid()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_feat_spatial,_ = torch.max(x, dim=1,keepdim=True)
        mean_feat_spatial = torch.mean(x, dim=1,keepdim=True)
        att_feat_spatial = torch.cat([max_feat_spatial, mean_feat_spatial], dim=1)

        att_weight_spatial = self.sigmoid(self.Conv(att_feat_spatial))
        x= x*att_weight_spatial
        return self.model(x)





