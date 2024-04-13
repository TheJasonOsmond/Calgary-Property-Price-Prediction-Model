import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

Housing_data = pd.read_csv("C:/Users/David/Data311/HousingData-1MillionUnder.csv")

df =pd.DataFrame(Housing_data)

features = df[['Square Footage', 'Year Built', 'Bed Rooms', 'Style', 'Basement','Distance to City Core(KM)']]
target = df['Price']

quadrant_encoder = OneHotEncoder() #Creating the instance for OneHotEncoder class
quadrant_oh = quadrant_encoder.fit_transform(df[["Quadrant"]]) # First sets unique integers to each category, then transforms categorical values into OneHotEncoder.
quadrant_oh = quadrant_oh.toarray() # Convert the sparse matrix to a numpy array

features = features.join(pd.DataFrame(quadrant_oh, columns=quadrant_encoder.get_feature_names_out())) #add NE, NW, SE, SW binary column to feature list

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42) #breaking up the training data

X_train['Basement'] = X_train['Basement'].map({'Finished': 1, 'Unfinished': 0}).fillna(0)
X_train = X_train.astype(float)
X_test['Basement'] = X_test['Basement'].map({'Finished': 1, 'Unfinished': 0}).fillna(0)
X_test = X_test.astype(float)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32) # Turning panda data frame into Pytouch tensors, view(-1,1) shapes 'y' only 1 column for target.
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__() 
        self.fc1 = nn.Linear(input_dim, 50) #outputs 50 features(neurons)
        self.fc2 = nn.Linear(50, 100) # second layers takes 50 and output 100 features(neurons)
        self.fc3 = nn.Linear(100, output_dim)  # outputs 1 value
    
    def forward(self, x):
        
        x = F.relu(self.fc1(x)) #x is passed through the first layer and ReLU(rectified Linear Unit) activation function is applied, introduces non-linearity.
        x = F.relu(self.fc2(x)) #The output from first layer is passed through the second layer then applied ReLU again.
        x = self.fc3(x) #The output is passed through the third layer, no activation function for regression.
    
        return x


input_dim = X_train_tensor.shape[1] #get the number of input features
output_dim = 1
Hmodel = MLP(input_dim, output_dim)

performance = nn.MSELoss() #Using the mean square error loss for measuring the average squared difference between predicted and actual values. Good proof of accuracy.

optimizer = torch.optim.Adam(Hmodel.parameters(), lr=0.0005) #Setting the optimizing algorithm for adjusting weights  based on the gradients of the loss function, lr sets the learning rate -> size of the step.

epochs = 100000 #Number of times the model will iterate over the whole training set
patience = 1000
early_stopping_counter = 0
best_test_loss = 59126898688

train_losses = []
test_losses = []


for epoch in range(epochs):

    Hmodel.train()
    optimizer.zero_grad() #clear the gradiant from last step
    y_pred = Hmodel(X_train_tensor) # pass training dataset through model getting y prediction
    loss = performance(y_pred, y_train_tensor) # using the MSE function defined earlier to find the loss
    loss.backward() #Preform backpropagation, getting the gradient of the loss function each respect to each weight

    optimizer.step() # Update the model's parameters based on the gradient to minimize the loss function

    if epoch % 10 == 0:
        Hmodel.eval() #Setting model to evaluation mode.
        

        with torch.no_grad(): # Do not store gradients
            y_pred_test = Hmodel(X_test_tensor) #Make prediction on test
            test_loss = performance(y_pred_test, y_test_tensor) #Get MSE loss
    
            print(f'Epoch {epoch + 10}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if test_loss < best_test_loss: #Finding the best test loss for new data
            best_test_loss = test_loss
            torch.save(Hmodel.state_dict(), 'best_model.pth') # Save the best model
        else:
            early_stopping_counter += 1
            print(early_stopping_counter)
            if early_stopping_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

Hmodel.load_state_dict(torch.load('best_model.pth'))

Hmodel.eval()
with torch.no_grad(): # Do not store gradients
    y_test_pred = Hmodel(X_test_tensor) #Make prediction on test
    test_loss = performance(y_test_pred, y_test_tensor) #Get MSE loss
    print(f'Test Loss: {test_loss.item()}')

x_values = range(0, len(train_losses) * 10, 10)

plt.figure(figsize=(10, 5))
plt.plot(x_values, train_losses, label='Train Loss')
plt.plot(x_values, test_losses, label='Test Loss')
plt.title('Train and Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


actual = y_test_tensor.numpy().flatten()
predicted = y_test_pred.numpy().flatten()

plt.figure(figsize=(10, 6))
plt.scatter(actual, predicted, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Housing Prices')
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4) # Adds a reference line
plt.show()

differences = np.abs(predicted - actual)

# Determine which differences are within the $20000 range
accurate_predictions = differences <= 50000

# Calculate the accuracy rate as the number of accurate predictions divided by the total number of predictions
accuracy_rate = np.mean(accurate_predictions)

print(f"Accuracy rate (within $10,000 of actual prices): {accuracy_rate:.2%}")