import torch
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset

# Load pre-trained ResNet-101 model
torch_model = models.resnet101(weights='IMAGENET1K_V1')
torch_model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize with mean and std for ImageNet
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]
    )
])

# Load the ImageNet dataset
dataset = load_dataset('imagenet-1k', split='train', streaming=True)

# Get 1000 images from the dataset for inference
imgs = list(dataset.take(1000))

count = 0
# Perform inference on each image
for v in imgs:
    try: 
        image = v['image']
        label = v['label']
        
        # Apply the transformation
        image_tensor = transform(image)

        # Add a batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
        print(image_tensor.shape)

        # Perform inference
        with torch.no_grad():
            output = torch_model(image_tensor)  # Forward pass

        # Get the predicted class index
        max_index = output.argmax(dim=1).item()  # Get the index of the max log-probability
        if max_index == label:
            print(max_index)
            count += 1
            print("Correct prediction!" + str(count))
    except:
        print("oops")
    
