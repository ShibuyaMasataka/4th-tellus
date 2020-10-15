import os
import os.path as osp

import torch
import torchvision

from models.pose_hrnet import get_pose_net
from models.utils import load_model, save_model
from data.dataset import Phase1Dataset, get_train_valid_split_data_names
from utils.loss import HMLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from opts import opts
from utils.visualize import visualization

# 学習用関数
def train(train_loader, model, optimizer, criterion, device, total_epoch, epoch):
    model.train() # モデルを学習モードに変更
    accumulate_datas = []
    # ミニバッチごとに学習
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        img_path = data["img_path"]
        inputs = data["input"].to(device) # GPUを使用するため，to()で明示的に指定
        anos = data["hm"].to(device) # GPUを使用するため，to()で明示的に指定

        # zero the parameter gradients
        optimizer.zero_grad() # 勾配を初期化

        # forward + backward + optimize
        outputs = model(inputs) # 順伝播の計算
        loss = criterion(outputs, anos) # 誤差を計算
        loss.backward() # 誤差を逆伝播させる
        optimizer.step() # 重みを更新する

        print("\rEpoch [%d/%d], Iterate : %d, Loss : %.4f" % (epoch,total_epoch,i,loss.item()), end='')
        accumulate_datas.append({"img_name":osp.basename(img_path[0])[:-4], "img":inputs[0], "ano":anos[0], "output":outputs['hm'][0]})

    print("")
    return accumulate_datas

def valid(valid_loader, model, criterion, device):
    model.eval() # モデルを推論モードに変更
    accumulate_loss = 0 # 平均loss計算用の変数を宣言

    accumulate_datas = []

    # ミニバッチごとに推論
    with torch.no_grad(): # 推論時には勾配は不要
        for i, data in enumerate(valid_loader, 0):
            # get the inputs
            img_path = data["img_path"]
            inputs = data["input"].to(device) # GPUを使用するため，to()で明示的に指定
            anos = data["hm"].to(device) # GPUを使用するため，to()で明示的に指定

            outputs = model(inputs) # 順伝播の計算

            accumulate_loss += criterion(outputs, anos) # 誤差を計算
            accumulate_datas.append({"img_name":osp.basename(img_path[0])[:-4], "img":inputs[0], "ano":anos[0], "output":outputs['hm'][0]})

    # 平均lossを計算
    data_num = len(valid_loader.dataset) # テストデータの総数
    mean_loss = accumulate_loss / data_num
    print("Validation Loss : %.4f" % (mean_loss))
    return mean_loss, accumulate_datas

def main(opt):
    train_data, valid_data = get_train_valid_split_data_names(opt.img_folder, opt.ano_folder, valid_size=1/8)

    # データの読み込み
    print("load data")
    train_dataset = Phase1Dataset(train_data, load_size=(640, 640), augment=True)
    print("train data length : %d" % (len(train_dataset)))
    valid_dataset = Phase1Dataset(valid_data, load_size=(640, 640), augment=False)
    print("valid data length : %d" % (len(valid_dataset)))
    # DataLoaderの作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # GPUの設定（PyTorchでは明示的に指定する必要がある）
    device = torch.device('cuda' if opt.gpus > 0 else 'cpu')

    # モデルの作成
    heads = {'hm': 1}
    model = get_pose_net(18, heads, 256).to(device)
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer)

    # 最適化手法を定義
    #optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)#, momentum=m, dampening=d, weight_decay=w, nesterov=n)
    # 損失関数を定義
    criterion = HMLoss()
    # 学習率のスケジューリングを定義
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)

    start_epoch = 0
    best_validation_loss = 1e10
    # 保存用フォルダの作成
    os.makedirs(os.path.join(opt.save_dir, opt.task, 'visualized'), exist_ok=True)

    # 学習 TODO エポック終了時点ごとにテスト用データで評価とモデル保存
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print("learning rate : %f" % scheduler.get_last_lr()[0])
        accumulate_datas = train(train_loader, model, optimizer, criterion, device, opt.num_epochs, epoch)
        visualization(os.path.join(opt.save_dir, opt.task, 'visualized'),
                    accumulate_datas)
        scheduler.step()

        # 最新モデルの保存
        save_model(os.path.join(opt.save_dir, opt.task, 'model_last.pth'),
                   epoch, model, optimizer, scheduler)

        # テスト用データで評価
        validation_loss, accumulate_datas = valid(valid_loader, model, criterion, device)
        # ベストスコア更新でモデルの保存
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_model(os.path.join(opt.save_dir, opt.task, 'model_best.pth'),
                       epoch, model, optimizer, scheduler)
            print("saved best model")
            visualization(os.path.join(opt.save_dir, opt.task, 'visualized'),
                        accumulate_datas)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().parse()
    main(opt)
