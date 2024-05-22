import torch
import torch.nn as nn
from torch import optim

from nflows import flows, distributions

from transform import create_transform
import json
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import math
from utils import *





class NFOptimizer_cs(nn.Module):
	def __init__(self,args,perc_k):
		super().__init__()
		self.args=args
		self.perc_k=perc_k
		self.img_dim=3*64*64
		self.latent_dim=3*64*64
		device=args.device
		with open(self.args.flow_config) as fp:
			flow_config = json.load(fp)
		distribution = distributions.StandardNormal((self.img_dim,))
		transform = create_transform(3, 64, 64,
	                                 num_bits=8,
	                                 **flow_config)
		gen = flows.Flow(transform, distribution)
		gen.load_state_dict(torch.load(args.ckpt, map_location=device))
		gen.eval()

		self.gen=gen.to(device)
		self.init_state()

	def init_state(self):
		self.best=None

		self.perc_k=self.perc_k/100


		k_dims=int(self.perc_k*self.latent_dim)

		self.latent_z = torch.zeros((self.args.batchsize,self.img_dim),
	                        dtype=torch.float,
	                        requires_grad=True, device=self.args.device)

		mask_indices = torch.tensor([k_dims])[:, None]
		mask = (reversed(torch.arange(self.img_dim)).expand(mask_indices.shape[0], -1)
                < mask_indices).float()
		self.mask = mask.to(self.args.device) 

	def invert_(self,y,img,A,gamma):
		self.img=img
		self.y=y
		self.A=A
		optimizer=optim.Adam([self.latent_z],self.args.lr)
		pbar = tqdm(range(self.args.steps))
		mse_min = np.inf
		mse_loss = 0
		reference_loss = 0
		
		
		for i in pbar:
			
			loss=0
			latent_z = (self.latent_z * self.mask[None, ...]).permute([1, 0, 2]).reshape(-1, self.img_dim)

			
			img_gen,log_det=self.gen._transform.inverse(latent_z)
			log_det=torch.abs(log_det)

			gen_obsv=torch.matmul(img_gen.view([-1,self.img_dim]),self.A)
			mse_loss = F.mse_loss(self.y,gen_obsv).mean()
			
			post_loss=(latent_z.norm(dim=1)**2  + log_det)
			post_loss=gamma*nats_to_bits_per_dim(post_loss,3,64,64)

			
			reference_vector = self.img 

			loss+=mse_loss + post_loss.mean()
			
			reference_loss = F.mse_loss(reference_vector,img_gen).mean()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if  reference_loss < mse_min:
				mse_min = reference_loss
				self.best = img_gen.detach().cpu()

			pbar.set_description(
                (
                    f" mse: {mse_loss:.4f};"
                    f" post: {post_loss.mean():.4f};"
                    f" mse_ref: {reference_loss:.4f};"
                )
            )

	def invert(self,y,img,A,gamma):
		self.invert_(y,img,A,gamma)
		return self.img,self.best




def make_measurements_noise(args,img,perc_m):
	img_dim=64*64*3
	n=64*64*3

	test_img=img.view([-1,img_dim])

	perc=perc_m/100
	
	m=int(perc*(img_dim))

	
	A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
	A = torch.tensor(A, dtype=torch.float, requires_grad=False, device=args.device)
	
	noise = np.random.normal(0,1,size=(args.batchsize,m))
	noise = noise * 0.1/np.sqrt(m)

	noise*=255
	noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=args.device)

	y=torch.matmul(test_img,A) + noise

	return y,A

