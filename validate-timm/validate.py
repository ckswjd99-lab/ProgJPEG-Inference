import timm
import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import get_imnet1k_dataloader

BATCH_SIZE = 128
IMNET_DIR = '/data/imagenet'


train_loader, val_loader = get_imnet1k_dataloader(root=IMNET_DIR, batch_size=BATCH_SIZE, augmentation=False)

print("Loaded ImageNet-1k dataset")

model_list = [
    'tiny_vit_5m_224.dist_in22k_ft_in1k',
]

model_vloss = []
model_vacc = []
model_nparams = []

criterion = nn.CrossEntropyLoss().cuda()



# validation
@torch.no_grad()
def validate(test_loader, model, criterion):
    model.eval()
    
    num_data = 0
    num_correct = 0
    sum_loss = 0
    
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for data, target in pbar:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        
        _, predicted = torch.max(output.data, 1)
        num_data += target.size(0)
        num_correct += (predicted == target).sum().item()
        sum_loss += loss.item() * target.size(0)

        avg_loss = sum_loss / num_data
        accuracy = num_correct / num_data

        pbar.set_description(f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    avg_loss = sum_loss / num_data
    accuracy = num_correct / num_data
    return avg_loss, accuracy


for mname in model_list:
    model = timm.create_model(mname, pretrained=True, num_classes=1000).cuda()
    nparams = sum(p.numel() for p in model.parameters())

    val_loss, val_acc = validate(val_loader, model, criterion)

    model_vloss.append(val_loss)
    model_vacc.append(val_acc)

    num_params = sum(p.numel() for p in model.parameters())
    model_nparams.append(num_params)

    print(f'|  {mname:44s} ({nparams:11,d}) |  {val_loss:.4f}  |  {val_acc*100:.2f} %  |')
