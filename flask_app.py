from flask import Flask, request, url_for, redirect, send_file, render_template,jsonify
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

app = Flask(__name__)   

#to upload file
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(81 * 32 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 3)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 81 * 32 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)        
        return x



def loadTrainData(train=False,root_dir=None):
    results=[]
    numOfTest = 1

    # Test
    for index in range(numOfTest):
        im = Image.open(root_dir)
        results.append({'x': im, 'y': torch.tensor(0)})
        

    return results
    



class CustomDataset(Dataset):
    def __init__(self, train=True, transform=None,root_dir=None):

        self.samples = loadTrainData(train=train,root_dir=root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.transform(self.samples[index]['x']), self.samples[index]['y']



train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])








@app.route('/', methods = ['GET','POST'])
def insert_data():

    
    if request.method == 'POST':

        uploaded_file = request.files['file']
        print("XXXXXXXXXXXXXXXXfile name: ",uploaded_file.filename)


        if uploaded_file.filename.rsplit('.', 1)[-1].lower()=='jpg' or uploaded_file.filename.rsplit('.', 1)[-1].lower()=='png' or uploaded_file.filename.rsplit('.', 1)[-1].lower()=='jpeg':
            if uploaded_file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)

                uploaded_file.save(file_path)

                root_dir = file_path
                print('FILE PATH: ',file_path)
                test_dataset = CustomDataset(train=False, transform=train_transforms, root_dir=root_dir)

                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

                # load the model
                import __main__
                setattr(__main__, "Net", Net)
                model = torch.load("./model_80.pt", map_location=torch.device("cpu"))
                
                
                
                
                
                model.eval()
                # prediction
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs, labels
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                print(predicted)

                if predicted == torch.tensor([0]):
                         pred = 'Bed'
                if predicted == torch.tensor([1]):
                         pred = 'Chair'
                if predicted == torch.tensor([2]):
                         pred = 'Sofa'
                return render_template('insert_data.html', message=f"The image is a {pred}")
                
                
            else:
                return render_template('insert_data.html', message="The file does not have any name") 

        else:
            return render_template('insert_data.html', message="The file format is not JPG/PNG")
       
        

    return render_template('insert_data.html', message="Please upload your image")

if __name__=='__main__':

    app.run(host='127.0.0.1',port=8000,debug=True)

    

    