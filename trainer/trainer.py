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

        self.train_metrics = MetricTracker('loss','loss_img','loss_mask','loss_edge','loss_flat','loss_lap','loss_cd', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (data, target, mask) in enumerate(tqdm(self.data_loader)):
            data = [item.to(self.device) for item in data]
            mask = [item.to(self.device) for item in mask]
            self.optimizer.zero_grad()
            output,rec_mesh,img_probs,faces,new_mesh,input_texture = self.model(data)
            loss_img = 0
            loss_mask = 0
            loss_lap = 0
            loss_edge = 0
            loss_flat = 0
            loss_cd = 0
            for i in range(VIEW_NUMS):
                img = output[i]
                # colored image L1 loss
                loss_img += L1(img, data[i])
                # 轮廓mask IOU L1/L2
                loss_mask += L1(img_probs[i],mask[i])
            # Lap平滑损失
            loss_lap+=mesh_laplacian_smoothing(new_mesh)
            # 边长损失
            loss_edge+=mesh_edge_loss(new_mesh)
            # 法向损失
            loss_flat+=mesh_normal_consistency(new_mesh)
            # CD损失
            # for i in range(rec_mesh.shape[0]):
            #     # 生成物体和参考圆柱之间的CD损失
            #     loss_cd += kal.metrics.point.chamfer_distance(mesh_trans.vertices,rec_mesh[i])
            loss_mask/=VIEW_NUMS
            loss_img/=VIEW_NUMS
            loss_cd/=VIEW_NUMS
            loss = loss_img+loss_mask+loss_lap+loss_edge+loss_flat+loss_cd
            loss.backward()
            self.optimizer.step()
            # log
            if batch_idx % self.log_step == 0:
                # 写入当前step
                step = (epoch - 1) * self.len_epoch + batch_idx
                self.writer.set_step(step)
                # 写入损失曲线
                if type(loss_img) == type(loss): self.train_metrics.update('loss_img', loss_img.item())
                if type(loss_mask) == type(loss): self.train_metrics.update('loss_mask', loss_mask.item())
                if type(loss_lap) == type(loss): self.train_metrics.update('loss_lap', loss_lap.item())
                if type(loss_edge) == type(loss): self.train_metrics.update('loss_edge', loss_edge.item())
                if type(loss_flat) == type(loss): self.train_metrics.update('loss_flat', loss_flat.item())
                if type(loss_cd) == type(loss): self.train_metrics.update('loss_cd', loss_cd.item())
                self.train_metrics.update('loss', loss.item())
                # 合成两张图像
                shape = data[0].shape
                input_img = torch.zeros([6,shape[1],shape[2],shape[3]])
                output_img = torch.zeros([6,shape[1],shape[2],shape[3]])
                # tb显示图像
                for i in range(6):
                    input_img[i] = data[i][0].cpu()
                    output_img[i] = output[i][0].cpu().detach()
                self.writer.add_image('input', make_grid(input_img, nrow=6, normalize=False))
                self.writer.add_image('output', make_grid(output_img, nrow=6, normalize=False))
                # 写入uvmap
                self.writer.add_image('uvmap', make_grid(input_texture[0].cpu().detach().unsqueeze(0), nrow=1, normalize=False))
                # 控制台log
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # 保存为三维模型, point写入obj文件, face固定的, uv坐标值
                save_mesh(rec_mesh[0].cpu().detach(),faces.long().cpu().detach(),os.path.join(self.config.obj_dir,'{}_{}_{}.obj'.format(epoch,batch_idx,step)))
                # exit()
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        self.do_validation = False
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
        with torch.no_grad():
            if self.veri_mode == False:
                for batch_idx, (data, target) in enumerate(tqdm(self.valid_data_loader)):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            else:
                distances = []
                distance_data_list = []
                labels = []
                for batch_idx, (data1,data2, target) in enumerate(self.valid_data_loader):
                    data1,data2, target = data1.to(self.device),data2.to(self.device), target.to(self.device)

                    output1 = self.model.extract_feature(data1)
                    output2 = self.model.extract_feature(data2)
                    dis = F.cosine_similarity(output1, output2).cpu()
                    distances.append(dis)
                    distance_data_list.append(np.array(dis))
                    labels.append(target)
                    
                # cat all distances
                distances = torch.cat(distances)
                # cat all labels
                label = torch.cat(labels)
                # cal eer
                intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final,eer, bestThresh, minV = calc_eer(distances, label)
                self.logger.debug('eer : {}, bestThresh : {},'.format(eer,bestThresh))
                self.logger.debug("intra_cnt is : {} , inter_cnt is {} , intra_len is {} , inter_len is {}".format(intra_cnt_final,inter_cnt_final,intra_len_final,inter_len_final))
                self.writer.set_step((epoch - 1), 'valid')
                self.valid_metrics.update('loss', eer)
                self.writer.add_image('input', make_grid(data1.cpu(), nrow=8, normalize=True))
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
