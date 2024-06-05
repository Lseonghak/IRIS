import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask values are within the range [0, 1]
        mask = np.clip(mask, 0, 1)
        
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

images_dir = 'sidewalk_unet/images'
masks_dir = 'sidewalk_unet/masks'
full_dataset = CrackDataset(images_dir, masks_dir, transform=None)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) 

num_classes = 2
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1)) 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 10
scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Training

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, masks)

            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")

torch.save(model.state_dict(), './checkpoints/deeplabv3_finetuned.pth')

def save_predictions(images, masks, outputs, output_dir='predictions'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, (image, mask, output) in enumerate(zip(images, masks, outputs)):
        image = image.permute(1, 2, 0).cpu().numpy()
        mask = mask.cpu().numpy()
        output = torch.argmax(output, dim=0).cpu().numpy()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[1].imshow(mask)
        axes[1].set_title('Mask')
        axes[2].imshow(output)
        axes[2].set_title('Prediction')

        plt.savefig(os.path.join(output_dir, f'prediction_{i}.png'))
        plt.close(fig)

model.eval()
images, masks = next(iter(val_loader))
images, masks = images.to(device), masks.to(device)
outputs = model(images)['out']
save_predictions(images, masks, outputs)
