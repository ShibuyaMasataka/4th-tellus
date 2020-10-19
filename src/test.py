import os
import os.path as osp

import torch
import torchvision

from models.pose_hrnet import get_pose_net
from models.utils import load_model, save_model
from data.dataset import TestDataset, get_test_data_names
from utils.loss import HMLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from opts import opts
from utils.visualize import visualization
from utils.submit import save_json
from models.decode import hm_decode

def test(test_loader, model, device):
    model.eval() # モデルを推論モードに変更

    # ミニバッチごとに推論
    accumulate_datas = []
    with torch.no_grad(): # 推論時には勾配は不要
        for i, data in enumerate(test_loader, 0):
            # get the inputs
            img_path = data["img_path"]
            inputs = data["input"].to(device) # GPUを使用するため，to()で明示的に指定
            ratio = data["ratio"][0]
            padw = data["padw"][0]
            padh = data["padh"][0]
            oimg = data["oimg"][0]


            outputs = model(inputs) # 順伝播の計算

            detections, heat = hm_decode(outputs['hm'], 0.1, K=10000)
            points = detections[:, :2]
            #points = points * 4

            points[:, 0] = (points[:, 0] - padw) / ratio
            points[:, 1] = (points[:, 1] - padh) / ratio

            accumulate_datas.append({"img_name":osp.basename(img_path[0])[:-4], "img":inputs[0], "output":heat[0], "points":points, "oimg":oimg})

    return accumulate_datas

def main(opt):
    test_data = get_test_data_names(opt.img_folder)

    # データの読み込み
    print("load data")
    test_dataset = TestDataset(test_data, load_size=(640, 640), limit=opt.limit)
    print("test data length : %d" % (len(test_dataset)))
    # DataLoaderの作成
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    # GPUの設定（PyTorchでは明示的に指定する必要がある）
    device = torch.device('cuda' if opt.gpus > 0 else 'cpu')

    # モデルの作成
    heads = {'hm': 1}
    model = get_pose_net(18, heads, 256).to(device)

    checkpoint = torch.load(os.path.join(opt.save_dir, opt.task, 'model_best.pth'))
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)
    print('model loaded')

    # 保存用フォルダの作成
    os.makedirs(os.path.join(opt.save_dir, opt.task, 'test_output'), exist_ok=True)

    accumulate_datas = test(test_loader, model, device)

    visualization(os.path.join(opt.save_dir, opt.task, 'test_output'),
                       accumulate_datas)
    save_json(os.path.join(opt.save_dir, opt.task), accumulate_datas)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().parse()
    main(opt)
