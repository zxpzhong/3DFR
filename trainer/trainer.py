import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker,calc_eer,save_mesh
import torch.nn.functional as F
from tqdm import tqdm
from loss.loss import L1,L2,Lap_Loss,CE,Edge_regularization,Loss_flat
import os
import kaolin as kal

# pytorch3d loss
from pytorch3d.loss.mesh_laplacian_smoothing import mesh_laplacian_smoothing
from pytorch3d.loss.mesh_edge_loss import mesh_edge_loss
from pytorch3d.loss.mesh_normal_consistency import mesh_normal_consistency

VIEW_NUMS = 6

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None,test_data_loader = None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 50*int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss','loss_cls','acc', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss','valid_eer','test_eer', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        sum_ = 0
        right_ = 0
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            _,output = self.model(data)
            # Compute accuracy
            pred_label = torch.argmax(output, dim=1)
            right_ += torch.sum((pred_label == target).float()).cpu().item()
            sum_+=data.shape[0]
            # 点云的分类损失
            loss_cls = F.cross_entropy(output,target)
            loss = loss_cls
            loss.backward()
            self.optimizer.step()
            # log
            if batch_idx % self.log_step == 0:
                # 写入当前step
                step = (epoch - 1) * self.len_epoch + batch_idx
                self.writer.set_step(step)
                # 写入损失曲线
                if type(loss_cls) == type(loss): self.train_metrics.update('loss_cls', loss_cls.item())
                self.train_metrics.update('loss', loss.item())
                self.train_metrics.update('acc', right_/sum_)
                # 控制台log
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} ACC : {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    right_/sum_))
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        # self.do_validation = False
        if self.do_validation and epoch%self.config['trainer']['save_period'] == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        # 使用当前模型跑完整个验证集
        features = []
        with torch.no_grad():
            for batch_idx, (data) in enumerate(tqdm(self.valid_data_loader)):
                data = data.to(self.device)
                output,_ = self.model(data)
                output = output.cpu()
                for i in range(output.shape[0]):
                    features.append(output[i])
        
        # 使用遍历取得EER阈值
        # 构建distances和label列表
        # 根据数据集的查询表,构建对应的距离和标签
        distances = []
        label = []
        for item in self.valid_data_loader.dataset.query:
            dis = F.cosine_similarity(features[item[0]].unsqueeze(0), features[item[1]].unsqueeze(0))
            distances.append(dis)
            label.append(item[2])
        
        intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,valid_eer, bestThresh, minV = calc_eer(distances, label)
        self.logger.debug('valid_eer : {}, bestThresh : {},'.format(valid_eer,bestThresh))
        self.logger.debug("intra_cnt is : {} , inter_cnt is {} , intra_len is {} , inter_len is {}".format(intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final))
        
        # 遍历测试集,获取所有特征
        features = []
        with torch.no_grad():
            for batch_idx, (data) in enumerate(tqdm(self.test_data_loader)):
                data = data.to(self.device)
                output,_ = self.model(data)
                output = output.cpu()
                for i in range(output.shape[0]):
                    features.append(output[i])
        distances = []
        label = []
        for item in self.test_data_loader.dataset.query:
            dis = F.cosine_similarity(features[item[0]].unsqueeze(0), features[item[1]].unsqueeze(0))
            distances.append(dis)
            label.append(item[2])
        intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,test_eer, bestThresh, minV = calc_eer(distances, label ,[bestThresh])
        self.logger.debug('test_eer : {}, bestThresh : {},'.format(test_eer,bestThresh))
        self.logger.debug("intra_cnt is : {} , inter_cnt is {} , intra_len is {} , inter_len is {}".format(intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final))
        
        # tensorboard操作
        self.writer.set_step((epoch - 1), 'valid')
        self.valid_metrics.update('valid_eer', valid_eer)
        self.valid_metrics.update('test_eer', test_eer)
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
