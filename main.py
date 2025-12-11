import time
from loss import *
from models import MLP
from utils import *
import torch 
import numpy as np
from sklearn.model_selection import KFold
import argparse
from scipy import stats
import sys
import os
import csv
import pandas as pd
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'test-data', 'bike'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'GAR experiments')
  parser.add_argument('--loss', default='GAR', type=str, help='loss functions to use ()')
  parser.add_argument('--dataset', default='wine_quality', type=str, help='the name for the dataset to use')
  parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
  parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter for SGD optimizer')
  parser.add_argument('--decay', default=1e-4, type=float, help='weight decay for training the model')
  parser.add_argument('--batch_size', default=256, type=int, help='training batch size')
  parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train')
  parser.add_argument('--noise', default=0.0, type=float, help='noise level for noisy datasets (0, 10, 30, 50)')

  # paramaters
  args = parser.parse_args()
  SEED = 123
  BATCH_SIZE = args.batch_size
  lr = args.lr
  decay = args.decay
  set_all_seeds(SEED)
  # dataloader
  num_targets = 1
  noise = args.noise

  if args.dataset == 'bike_sharing':
      if noise in [10, 30, 50]:
          path = f'advanced-data/{noise}/bike_sharing_noisy.npz'
          trX, trY, teX, teY = bike_sharing(path=path)
      else:
          trX, trY, teX, teY = bike_sharing()
      num_targets = trY.shape[1]

  elif args.dataset == 'protein':
      if noise in [10, 30, 50]:
          path = f'advanced-data/{noise}/protein_noisy.npz'
          trX, trY, teX, teY = protein_data(path=path)
      else:
          trX, trY, teX, teY = protein_data()
      num_targets = trY.shape[1]

  elif args.dataset == 'sine':
      if noise in [10, 30, 50]:
          # noise folder maps to noise level: 10 → 0.1, 30 → 0.3, 50 → 0.5
          noise_level = noise / 100.0
          path = f'advanced-data/{int(noise)}/sine_noisy_scale5_noise{noise_level}.npz'

          trX, trY, teX, teY = sine_data(path=path, noise_level=noise_level, split_data=True)
      else:
          trX, trY, teX, teY = sine_data(split_data=True)

  print(trX.shape)
  print(teX.shape)
  tr_pair_data = pair_dataset(trX, trY)
  te_pair_data = pair_dataset(teX, teY)
  testloader = torch.utils.data.DataLoader(dataset=te_pair_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)

  testloader = torch.utils.data.DataLoader(dataset=te_pair_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)
  
  epochs = args.epochs
  milestones = [int(epochs*0.5), int(epochs*0.75)]

  kf = KFold(n_splits=5)
  tmpX = np.zeros((trY.shape[0],1))
  part = 0
  
  # Initialize results storage
  results_list = []
  epoch_results_list = []
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  results_file = f'results_{args.dataset}_{args.loss}_{timestamp}.csv'
  epoch_results_file = f'epoch_results_{args.dataset}_{args.loss}_{timestamp}.csv'
  
  print ('Start Training')
  print ('-'*30)
  total_start_time = time.time()

  paraset = [0.1, 0.5, 0.9]
  if args.loss in ['Huber', 'focal-MAE', 'focal-MSE']:
    paraset = [0.25,1,4]
  elif args.loss in ['MAE', 'MSE']:
    paraset = [0.1,0.5,0.9] # dummy repeats
  elif args.loss == 'ranksim':
    paraset = [0.5,1,2]
  elif args.loss in ['GAR', 'GAR-EXP']:
    paraset = [0.1, 1, 10]
  elif args.loss in ['RNC']:
    paraset = [1,2,4]
  elif args.loss in ['ConR']:
    paraset = [0.2,1,4]

  # Set device - prefer XPU (Intel XE Graphics), then CUDA, then CPU
  if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
    print(f'Using Intel XE Graphics (XPU): {torch.xpu.get_device_name(0)}')
  elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA GPU: {torch.cuda.get_device_name(0)}')
  else:
    device = torch.device('cpu')
    print('Using CPU')
  
  # Initialize gradient data storage
  gradient_data = {para: [] for para in paraset}

  for train_id, val_id in kf.split(tmpX):
    tmp_trainSet = torch.utils.data.Subset(tr_pair_data, train_id)
    tmp_valSet = torch.utils.data.Subset(tr_pair_data, val_id)
    for para in paraset: 
      trainloader = torch.utils.data.DataLoader(dataset=tmp_trainSet, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
      validloader = torch.utils.data.DataLoader(dataset=tmp_valSet, batch_size=BATCH_SIZE, num_workers=0, shuffle=False, drop_last=False)
      basic_loss = torch.nn.L1Loss()
      if args.dataset in ['abalone', 'wine_quality', 'CCS', 'CCPR', 'PM25']:
        model = MLP(input_dim=trX.shape[-1], hidden_sizes=(16,32,16,8, ), num_classes=num_targets).to(device)
      elif args.dataset in ['supercon', 'parkinson-motor', 'parkinson-total', 'IC50', 'MITV']:
        model = MLP(input_dim=trX.shape[-1], hidden_sizes=(128,256,128,64, ), num_classes=num_targets).to(device)
      elif args.dataset == 'bike_sharing':
        model = MLP(input_dim=trX.shape[-1], hidden_sizes=(64,128,64,32, ), num_classes=num_targets).to(device)
      elif args.dataset == 'protein':
        model = MLP(input_dim=trX.shape[-1], hidden_sizes=(64,128,64,32, ), num_classes=num_targets).to(device)
      else:
        model = MLP(input_dim=trX.shape[-1], hidden_sizes=(64,128,64,32, ), num_classes=num_targets).to(device)
      optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
      
      if args.loss in ['GAR', 'GAR-EXP']:
        basic_loss = GAR(alpha=para,version=args.loss)
      elif args.loss in ['MSE']:
        basic_loss = torch.nn.MSELoss()
      elif args.loss == 'Huber':
        basic_loss = torch.nn.HuberLoss(delta=para)
      elif args.loss == 'RNC':
        add_loss = RnCLoss(temperature=para)


      print('para=%s, part=%s'%(para, part))
      for epoch in range(epochs): # could customize the running epochs
        epoch_loss = 0
        pred = []
        truth = []
        start_time = time.time()
        for idx, data in enumerate(trainloader):
            optimizer.zero_grad()
            tr_X, tr_Y = data[0].to(device), data[1].to(device)
            pred_Y, feat = model(tr_X)
            pred.append(pred_Y.cpu().detach().numpy())
            truth.append(tr_Y.cpu().detach().numpy())
            if args.loss in ['GAR', 'GAR-EXP']:
              ratio = epoch/float(epochs)
              bloss = basic_loss(pred_Y, tr_Y)
              # Potentially can utilize adaptive alpha for GAR. We didn't use it in our experiments.
              # bloss = basic_loss(pred_Y, tr_Y, alpha = (0.1+ratio)*para)
            else:
              bloss = basic_loss(pred_Y, tr_Y)
            if args.loss in ['MAE', 'MSE', 'Huber', 'ranksim', 'focal-MAE', 'focal-MSE', 'ConR', 'GAR', 'GAR-EXP']:
              loss = bloss
            else:
              if args.loss in ['RNC']:
                aloss = add_loss(feat, tr_Y)
              else:
                aloss = add_loss(pred_Y, tr_Y)
            if args.loss == 'ranksim':
              loss += 100*batchwise_ranking_regularizer(feat, tr_Y, para)
            elif args.loss == 'ConR':
              if args.dataset in ['IC50']:
                loss += para*ConR_extend(feat, tr_Y, pred_Y)
              else:
                loss += para*ConR(feat, tr_Y, pred_Y)
            elif args.loss == 'focal-MAE':
              loss = weighted_focal_mae_loss(pred_Y, tr_Y, beta = para)
            elif args.loss == 'focal-MSE':
              loss = weighted_focal_mse_loss(pred_Y, tr_Y, beta = para)
            elif args.loss == 'RNC':
              if epoch < milestones[0]:
                loss = aloss
              else:
                loss = bloss
            epoch_loss += loss.cpu().detach().numpy()
            loss.backward()
            
            # Calculate gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient_data[para].append(total_norm)
            
            optimizer.step()
        scheduler.step()
        epoch_loss /= (idx+1)
        epoch_time = time.time() - start_time
        print('Epoch=%s, time=%.4f'%(epoch, epoch_time))

        preds = np.concatenate(pred, axis=0)
        truths = np.concatenate(truth, axis=0)
        MAE, RMSE, pearson, spearman = [], [], [], []
        for i in range(num_targets):
          pred, truth = preds[:,i], truths[:,i]
          MAE.append(np.abs(pred-truth).mean())
          RMSE.append(((pred-truth)**2).mean()**0.5)
          pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
          spearman.append(stats.spearmanr(truth, pred).statistic)
        print('Epoch=%s, train_loss=%.4f, train_MAE=%.4f, train_RMSE=%.4f, train_Pearson=%.4f, train_Spearman=%.4f, lr=%.4f'%(epoch, epoch_loss, np.mean(MAE), np.mean(RMSE), np.mean(pearson), np.mean(spearman), scheduler.get_last_lr()[0]))      
        
        pred = []
        truth = [] 
        model.eval()
        for idx, data in enumerate(validloader):
            te_X, te_Y = data[0].to(device), data[1].to(device)
            pred_Y, feat = model(te_X)
            pred.append(pred_Y.cpu().detach().numpy())
            truth.append(te_Y.cpu().detach().numpy())
        preds = np.concatenate(pred, axis=0)
        truths = np.concatenate(truth, axis=0)
        MAE, RMSE, pearson, spearman = [], [], [], []
        for i in range(num_targets):
          pred, truth = preds[:,i], truths[:,i]
          MAE.append(np.abs(pred-truth).mean())
          RMSE.append(((pred-truth)**2).mean()**0.5)
          pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
          spearman.append(stats.spearmanr(truth, pred).statistic)
        print('valid_MAE=%.4f, valid_RMSE=%.4f, valid_Pearson=%.4f, valid_Spearman=%.4f'%(np.mean(MAE), np.mean(RMSE), np.mean(pearson), np.mean(spearman)))

        pred = []
        truth = [] 
        for idx, data in enumerate(testloader):
            te_X, te_Y = data[0].to(device), data[1].to(device)
            pred_Y, feat = model(te_X)
            pred.append(pred_Y.cpu().detach().numpy())
            truth.append(te_Y.cpu().detach().numpy())
        preds = np.concatenate(pred, axis=0)
        truths = np.concatenate(truth, axis=0)
        MAE, RMSE, pearson, spearman = [], [], [], []
        for i in range(num_targets):
          pred, truth = preds[:,i], truths[:,i]
          MAE.append(np.abs(pred-truth).mean())
          RMSE.append(((pred-truth)**2).mean()**0.5)
          pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
          spearman.append(stats.spearmanr(truth, pred).statistic)
        test_mae, test_rmse, test_pearson, test_spearman = np.mean(MAE), np.mean(RMSE), np.mean(pearson), np.mean(spearman)
        print('test_MAE=%.4f, test_RMSE=%.4f, test_Pearson=%.4f, test_Spearman=%.4f'%(test_mae, test_rmse, test_pearson, test_spearman))
        
        # Store epoch results
        epoch_results_list.append({
            'fold': part,
            'parameter': para,
            'epoch': epoch,
            'epoch_time': epoch_time,
            'test_MAE': test_mae,
            'test_RMSE': test_rmse,
            'test_Pearson': test_pearson,
            'test_Spearman': test_spearman
        })
        
      # Store final results for this fold/parameter
      results_list.append({
          'fold': part,
          'parameter': para,
          'test_MAE': test_mae,
          'test_RMSE': test_rmse,
          'test_Pearson': test_pearson,
          'test_Spearman': test_spearman
      })
      model.train()

    part += 1 

  total_end_time = time.time()
  total_time = total_end_time - total_start_time
  print(f"\nTotal training time: {total_time:.2f} seconds")
  
  # Save final results to CSV
  if results_list:
    with open(results_file, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=['fold', 'parameter', 'test_MAE', 'test_RMSE', 'test_Pearson', 'test_Spearman'])
      writer.writeheader()
      writer.writerows(results_list)
    print('\n' + '='*50)
    print(f'Final results saved to: {results_file}')
    print('='*50)
  
  # Save final results to CSV (with total_time column)
  if results_list:
    # Thêm cột total_time vào header
    fieldnames = ['fold', 'parameter', 'test_MAE', 'test_RMSE', 'test_Pearson', 'test_Spearman', 'total_time']
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # thêm cột total_time rỗng cho từng dòng kết quả fold
        for row in results_list:
            row['total_time'] = ""
        writer.writerows(results_list)

        # ghi dòng cuối có thời gian tổng
        writer.writerow({'fold': '', 'parameter': '', 'test_MAE': '','test_RMSE': '', 'test_Pearson': '', 'test_Spearman': '','total_time': total_time})

    print('\n' + '='*50)
    print(f'Final results saved to: {results_file}')
    print('='*50)
  
  # Save epoch-by-epoch results to CSV
  if epoch_results_list:
    with open(epoch_results_file, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=['fold', 'parameter', 'epoch', 'epoch_time', 'test_MAE', 'test_RMSE', 'test_Pearson', 'test_Spearman'])
      writer.writeheader()
      writer.writerows(epoch_results_list)
    print(f'Epoch-by-epoch results saved to: {epoch_results_file}')
  
  # Save gradient data to CSV
  gradient_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in gradient_data.items()]))
  gradient_csv_file = f'gradient_norms_{args.dataset}_{args.loss}_{timestamp}.csv'
  gradient_df.to_csv(gradient_csv_file, index=False)
  print(f'Gradient norms saved to: {gradient_csv_file}')