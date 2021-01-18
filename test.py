import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import loss.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import os

def main(config):
    
    
    logger = config.get_logger('test')

    # setup data_loader instances
    train_data_loader = config.init_obj('train_data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)
    test_data_loader = config.init_obj('test_data_loader', module_data)
    
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, target, mask) in enumerate(tqdm(train_data_loader)):
            data = [item.to(device) for item in data]
            mask = [item.to(device) for item in mask]
            output,rec_mesh,img_probs,faces,new_mesh,input_texture = model(data)
            # save 3d model
            for i in range(data[0].shape[0]):
                model.meshtemp.export_obj(os.path.join(config.obj_dir,'{}.obj'.format(target[i])),rec_mesh[i],input_texture[i])
    with torch.no_grad():
        for batch_idx, (data, target, mask) in enumerate(tqdm(valid_data_loader)):
            data = [item.to(device) for item in data]
            mask = [item.to(device) for item in mask]
            output,rec_mesh,img_probs,faces,new_mesh,input_texture = model(data)
            # save 3d model
            for i in range(data[0].shape[0]):
                model.meshtemp.export_obj(os.path.join(config.obj_dir,'{}.obj'.format(target[i])),rec_mesh[i],input_texture[i])
    with torch.no_grad():
        for batch_idx, (data, target, mask) in enumerate(tqdm(test_data_loader)):
            data = [item.to(device) for item in data]
            mask = [item.to(device) for item in mask]
            output,rec_mesh,img_probs,faces,new_mesh,input_texture = model(data)
            # save 3d model
            for i in range(data[0].shape[0]):
                model.meshtemp.export_obj(os.path.join(config.obj_dir,'{}.obj'.format(target[i])),rec_mesh[i],input_texture[i])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/home/zf/vscode/3d/DR_3DFM/saved/models/pytorch3duvmap/0118_005712_pytorch3d_imgL1maskL1lapedgeflat_uv/checkpoint-epoch85.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log', default=None, type=str,
                    help='log name')
    config = ConfigParser.from_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['n_gpu']
    main(config)
