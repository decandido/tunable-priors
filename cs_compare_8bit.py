import argparse
import os 
import pandas as pd
from skimage.measure import compare_psnr,compare_ssim,compare_mse
import warnings
from pathlib import Path
import json
import torch 
from inv_class_8bit import NFOptimizer_cs,make_measurements_noise
from data_hq import create_dataset
from torch.utils.data import DataLoader
import numpy as np


warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
if __name__ == "__main__":
	 parser = argparse.ArgumentParser(description='solve cs generated from trained NF')
	 parser.add_argument('-ckpt', type=str, help='which model to use',default='./nf_8bit/best_flow.pt')
	 parser.add_argument('-perc_m', type=int, help='which model to use',default=[15,10,7.5,5,2.5,1])
	 parser.add_argument('-device',type=str,help='device to use', default='cpu')
	 parser.add_argument('-perc_k',type=int, default=[100,15,10,7.5,5,2.5,1])
	 parser.add_argument('-gammas',type=float,default=[1.0])
	 parser.add_argument('-steps',type=int, default=300)
	 parser.add_argument('-image_size',type=int, default=64)
	 parser.add_argument('-batchsize',type=int, default=1)
	 parser.add_argument('-lr',type=float, default=.25)
	 parser.add_argument('--seed', type=int, default=42)
	 parser.add_argument('-lr_same_pace',type=bool,default=False)
	 parser.add_argument('-data_dir',type=str,default='./demo')
	 parser.add_argument('-save_folder',type=str,default='demo_run/')	
	 script_dir = Path(__file__).resolve().parent
	 parser.add_argument('--data_config', type=str,
                        default=script_dir / 'config_8bit' / 'data_config_celeba.json')
	 parser.add_argument('--flow_config', type=str,
                        default=script_dir / 'config_8bit' / 'flow_config.json')
	 args = parser.parse_args()

	 device=args.device

	 columns = ["K","M","gamma","PSNR","MSE","SSIM"]
	 dataframe = pd.DataFrame(columns=columns)
	 dataframe_deq=pd.DataFrame(columns=columns)
	 device=args.device
	 save_path=args.save_folder
	 if not os.path.exists(save_path):
	 	os.makedirs(save_path)

	 with open(args.data_config) as fp:
	 	data_config = json.load(fp)

	 test_dataset = create_dataset(root=args.data_dir,
                                      split='test',
                                      **data_config)
	 test_dataloader=DataLoader(test_dataset.dataset,
                             batch_size=args.batchsize,
                             shuffle=False)
	 

	 dict_exp={'k':[],'m':[],'gamma':[],'obs':[],'rec':[]}
	 dict_exp_deq={'k':[],'m':[],'gamma':[],'obs':[],'rec':[]}
	 torch.manual_seed(args.seed)
	 np.random.seed(args.seed)


	 for perc_m in args.perc_m:
	 	for i, img in enumerate(test_dataloader):
 			y,A=make_measurements_noise(args,img[0].to(device),perc_m)
 			for gamma in args.gammas:
 				for k in args.perc_k:
 					if perc_m==k or k==100:
			 			optim=NFOptimizer_cs(args,k)
			 			obs,gen=optim.invert(y,img[0].to(device),A,gamma)

			 			obs_np,gen_np=obs.cpu().numpy().transpose(0,2,3,1),gen.cpu().numpy().transpose(0,2,3,1)
			 			obs_deq=test_dataset.preprocess_fn.inverse(obs).detach().cpu().numpy().transpose(0,2,3,1)
			 			gen_deq=test_dataset.preprocess_fn.inverse(gen).detach().cpu().numpy().transpose(0,2,3,1)


			 			

					 	psnr=compare_psnr(obs_np[0], gen_np[0],data_range=255)
					 	ssim=compare_ssim(obs_np[0], gen_np[0],multichannel=True,data_range=255)
					 	mse=compare_mse(obs_np[0], gen_np[0])

					 	psnr_deq=compare_psnr(obs_deq[0], gen_deq[0])
					 	ssim_deq=compare_ssim(obs_deq[0], gen_deq[0],multichannel=True,data_range=1)
					 	mse_deq=compare_mse(obs_deq[0], gen_deq[0])

					 	print('*'*10,'PSNR Per image',psnr,'k',k,'*'*10)
					 	print('*'*10,'PSNR Per image',psnr_deq,'k',k,'*'*10)

					 	dict_exp['k'].append(k)
					 	dict_exp['m'].append(perc_m)
					 	dict_exp['gamma'].append(gamma)
					 	dict_exp['obs'].append(obs_np)
					 	dict_exp['rec'].append(gen_np)

					 	dict_exp_deq['k'].append(k)
					 	dict_exp_deq['m'].append(perc_m)
					 	dict_exp_deq['gamma'].append(gamma)
					 	dict_exp_deq['obs'].append(obs_deq)
					 	dict_exp_deq['rec'].append(gen_deq)

					 	np.save(save_path +"exps.npy",dict_exp)
					 	np.save(save_path +"exps_deq.npy",dict_exp_deq)

					 	dataframe=dataframe.append({"K":k,"M":perc_m,"gamma":gamma,"PSNR":psnr,"MSE":mse,"SSIM":ssim},ignore_index=True)
					 	dataframe_deq=dataframe_deq.append({"K":k,"M":perc_m,"gamma":gamma,"PSNR":psnr_deq,"MSE":mse_deq,"SSIM":ssim_deq},ignore_index=True)
					 	dataframe.to_csv(save_path + 'cs_k.csv')
					 	dataframe_deq.to_csv(save_path + 'cs_k_deq.csv')
					 	del optim,obs,gen






