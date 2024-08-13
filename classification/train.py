import glob
import os
import time
import torchvision
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from resnet2p1d import generate_model
from models import C3D_Light

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
root = "./wsd/"

train_dir = root + "sensor_data/train"
test_dir = root + "sensor_data/test"
writein = root + "results.txt"
modelsave = root + 'results/model'
# deform_idx = [] #str(deform_idx)
# identification of different models
p = "c3dL_"

output_re = open(writein, 'a')
output_re.write('\r\n')
output_re.write(train_dir + '\n')

# dataload
def load_lc_data(path, indices=0, split=False):
    txt_Paths = glob.glob(os.path.join(path, '*.txt'))
    txt_Paths.sort()
    sensor_data = []
    sensor_label = []

    for txt_item in txt_Paths:
        label = txt_item.split('-')[1]
        data_z = pd.read_csv(txt_item, delimiter=',', header=None)
        data_z = np.asarray(data_z)
        # print(data_z.shape)
        data = data_z.reshape((1, 22, 181, 180))

        # # sensor array optimization
        if indices.any():
        # if indices:
            data = data[0][indices]
            data = data.reshape((1, len(indices), 181, 180))

        # print(data.shape, type(data))
        if (label == 'health'):
            label_c = 0
        else:
            label_c = 1

        sensor_data.append(data)
        sensor_label.append(label_c)

    _data = np.array(sensor_data, dtype="float")
    _label = np.array(sensor_label, dtype="float")
    print(_data.shape)


    if split == True:
        # train,val-array,list
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            _data, _label,
            test_size=0.1,
            random_state=20,  # results can re
            shuffle=True,
            stratify=_label
        )
        return x_train, x_val, y_train, y_val
    else:
        return _data, _label


def main(args):

    ps = "m" + str(args['momentum']) + "[b" + str(args['batch']) + "e" + str(args['epochs']) + "i" + str(i) + "lr" + str(args['lr']) + p + str(args['optim_param'])+ "]"
    args['ps'] = ps


    ## ---------------------- model ---------------------- ##
    net = C3D_Light(num_classes=2, pretrained=False).to(device)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'], nesterov=True)
    loss_function = nn.CrossEntropyLoss()
    ## ---------------------- Dataload ---------------------- ##
    # optimized MTS after SAO
    optim_param = args['optim_param']
    indices = [3, 21, 9, 10, 14, 6, 0, 20, 7, 2, 13, 15, 12, 18, 4, 1, 16, 19, 8, 11, 17, 5]  # MAX
    indices = indices[:optim_param]
    indices.sort()

    # SIR
    index= [0,6,4,2,1,3,7,5]    # Max
    if index:
        indices = np.array(indices) #[ 0  1  4  7  8 12 13 16 20 21]
        re_indices = indices[index] #[ 7  4 12  0  8  1 16 20 21 13]
    print('re_indices: ',re_indices)
    output_re.write('re_indices: ' + str(re_indices) + '\t')
    indices = re_indices

    test_data, test_label = load_lc_data(test_dir,indices)
    x_train, x_val, y_train, y_val = load_lc_data(train_dir,indices, split=True)

    torch_train = torch.from_numpy(x_train)
    torch_valid = torch.from_numpy(x_val)
    torch_test = torch.from_numpy(test_data)

    train_num = len(torch_train)
    val_num = len(torch_valid)
    test_num = len(torch_test)
    print("using {} for training, {} for validation, {} for test.".format(train_num, val_num, test_num))

    torch_trainL = torch.tensor(y_train)
    torch_validL = torch.tensor(y_val)
    torch_testL = torch.tensor(test_label)

    train_ids = TensorDataset(torch_train, torch_trainL)
    valid_ids = TensorDataset(torch_valid, torch_validL)
    test_ids = TensorDataset(torch_test, torch_testL)

    train_loader = DataLoader(dataset=train_ids, batch_size=args['batch'], shuffle=True, num_workers=args['nw'])
    valid_loader = DataLoader(dataset=valid_ids, batch_size=args['batch'], shuffle=False, num_workers=args['nw'])
    test_loader = DataLoader(dataset=test_ids, batch_size=args['batch'], shuffle=False, num_workers=args['nw'])

    ## ---------------------- train ---------------------- ##
    train_curve = list()
    valid_curve = list()
    best_acc = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    train_steps = len(train_loader)  # num of iterations
    start.record()
    for epoch in range(args['epochs']):
        net.train()  # net is the model vgg16
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):
            imgs, targets = data
            imgs = imgs.float()
            optimizer.zero_grad()
            outputs = net(imgs.to(device))
            loss = loss_function(outputs, targets.long().to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_curve.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args['epochs'],
                                                                     loss)
        # scheduler.step()
        # print('lr={:.6f}'.format(scheduler.get_lr()[0]))

        # 3. validating
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(valid_loader)
            for val_data in val_bar:
                val_imgs, val_labels = val_data
                val_imgs = val_imgs.float()
                outputs = net(val_imgs.to(device))
                loss_val = loss_function(outputs, val_labels.long().to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        valid_curve.append(loss_val.item())

    # model save
    end.record()
    args['train_time'] = start.elapsed_time(end)
    output_re.write('train_time=' + str(args['train_time']) + '\t')
    print('train_time=', args['train_time'])
    train_x = range(len(train_curve))
    train_y = train_curve

    val_interval = 1
    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()

    print('VAL best: {0}, VAL end: {1}'.format(best_acc, val_accurate))
    # print(val_accurate)
    torch.save(net, modelsave + '/' + p + '-i' + str(i + 1) + 'end.pt')
    print('Finished Training')

    ## ---------------------- test ---------------------- ##
    output_re.write("end val acc: " + str(val_accurate) + '\n')
    # end model
    output_re.write("=====endmodel ACC===" + "=====\n")
    start.record()
    model2 = torch.load(modelsave + '/' + p + '-i' + str(i + 1) + 'end.pt')
    model2.eval()
    correct2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model2(data) #tensor([[-0.5464,  0.4531],[ 7.3246, -7.4213],...
            predict_2 = torch.max(output, dim=1)[1]  # tensor([1, 0, 0, 1, 1, 0, 1, 1])
            correct2 += torch.eq(predict_2, target.to(device)).sum().item()

    end.record()
    args['test_time'] = start.elapsed_time(end)
    output_re.write('test_time=' + str(args['test_time']) + '\t')
    print('test_time=', args['test_time'])
    print('\nEndmodel Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct2, len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset)))
    args['Acc'] = correct2 / len(test_loader.dataset)

    output_re.write('test:' + str(correct2 / len(test_loader.dataset)) + '\t')
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    output_re.write('time:' + str(t) + '\n')

if __name__ == '__main__':
    args = {'batch': 8, 'epochs': 200, 'lr': 0.0005,'momentum':0.9, 'nw': 8}
    N = 5
    avg_acc = 0
    avg_trtime = 0
    avg_tetime = 0

    lis = []
    op = 8
    for i in range(0, N):
        print("Turn {} for training.".format(i))
        args['optim_param'] = op
        main(args)
        print(args['Acc'])

        lis.append(args['Acc'])
        avg_acc += args['Acc']
        avg_trtime += args['train_time']
        avg_tetime += args['test_time']

        args['Acc'] = 0
        args['train_time'] = 0
        args['test_time'] = 0

    avg_acc = avg_acc / N
    avg_trtime = avg_trtime / N
    avg_tetime = avg_tetime / N

    print(lis)
    output_re.write('The average accuracy of ' + str(N) + ' times is ' + str(avg_acc) + '\n')
    output_re.write('The average training time of ' + str(N) + ' times is ' + str(avg_trtime) + '\n')
    output_re.write('The average testing time of ' + str(N) + ' times is ' + str(avg_tetime) + '\n')
    output_re.write('The best of ps' + args['ps'] + str(N) + ' are' + str(lis) + '\n')
