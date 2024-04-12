import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utilities import ResPINN


#Loading a Model

device = torch.device("gpu" if torch.cuda else "cpu")

loaded_model = ResPINN(input_size=2, output_size=7, n_hidden=4, hidden_width=100, activation=nn.Tanh(), res=False)

loaded = torch.load('models/fringe_model_width_100_depth_4.pth')

updated_state_dict = collections.OrderedDict()

for key, value in loaded.items():
    # Replace the key names to match your model's architecture
    new_key = key.replace("layers.", "")
    new_key = new_key.replace("net.4","net.8")
    new_key = new_key.replace("net.3","net.6")
    new_key = new_key.replace("net.2","net.4")
    new_key = new_key.replace("net.1","net.2")
    updated_state_dict[new_key] = value

loaded_model.load_state_dict(updated_state_dict)


def color_plot_outputs(model):
    #Color plots of the models output for all variables over time
    variables = ['rho', 'vx', 'vy', 'vz', 'By', 'Bz', 'p']

    val_t = torch.linspace(0, 0.2, 100, device=device) # time points to sample
    val_x = torch.linspace(-0.5, 0.5, 200, device=device) # position points to sample
    val_tx = torch.cartesian_prod(val_t, val_x) # combination of points

    # Extract t_points and x_points from val_tx
    t_points = val_tx[:, 0]
    x_points = val_tx[:, 1]

    # Get the model's output
    with torch.no_grad(): 
        val_u = loaded_model(t_points, x_points).squeeze().cpu().numpy()

    # Reshape the output tensor to organize the 7 variables
    val_u = val_u.reshape(len(val_t), len(val_x), 7)

    # Loop over the third dimension of val_u to create separate plots
    for i in range(7):
        plt.figure()
        im = plt.pcolormesh(val_t.cpu(), val_x.cpu(), val_u[:, :, i].T, shading="nearest", cmap="Spectral")
        plt.colorbar(im)
        plt.title(variables[i]) # using the variable names for the titles
        plt.xlabel("time")      # labeling the x-axis as "time"
        plt.ylabel("x")         # labeling the y-axis as "x"
        plt.tight_layout()
        plt.show()

color_plot_outputs(loaded_model)
# print(loaded_model)

# for name, param in loaded_model.named_parameters():
#      print(name, param)

#plotting the loss function of a model(call these nexttwo after training)

# Plot the loss history
# make sure converges to smaller loss
def plot_loss_history(loss_history):
    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss vs. Epoch for domain loss')
    plt.show()

# Plot the loss history divided up by contributions of different functions
def plot_loss_history_detailed(boundary_loss_history,initial_loss_history,domain_loss_history,fringe_loss_history):
    plt.figure()
    plt.plot(range(len(boundary_loss_history)), boundary_loss_history, label = "boundary")
    plt.plot(range(len(boundary_loss_history)), initial_loss_history, label = "initial_loss")
    plt.plot(range(len(boundary_loss_history)), domain_loss_history, label = "domain")
    plt.plot(range(len(boundary_loss_history)), fringe_loss_history, label = "fringe")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Loss vs. Epoch')
    plt.show()

def plot_model_predictions_at_time_point(model,inter_condition):
    model.eval()

    tplot= 0.15#0.147

    x = np.linspace(-0.5, 0.5, 100)
    t = np.linspace(tplot,tplot, 1)

    t, x = np.meshgrid(t, x)
    t, x = t.flatten(), x.flatten()
    t, x = torch.Tensor(t).to(device), torch.Tensor(x).to(device)
    numeric = inter_condition(tplot,x)
    numeric = numeric.reshape(100, 7)
    prediction = model(t, x).detach()
    prediction = prediction.reshape(100, 7)

    #plt.imshow(prediction)

    for i in range(7):

        plt.plot(x.cpu().numpy(), numeric[:,i].cpu().flatten().numpy())
        plt.plot(x.cpu().numpy(), prediction[:,i].cpu().flatten().numpy())
        plt.grid(True)
        plt.show()


#Example usage:
E_table = pd.read_csv('E.csv')
E_tensor = torch.tensor(E_table.values, dtype=torch.float32,device=device)
simt,simx=E_tensor.size()

By_table = pd.read_csv('By.csv')
p_table = pd.read_csv('p.csv')
rho_table = pd.read_csv('rho.csv')
vx_table = pd.read_csv('vx.csv')
vy_table = pd.read_csv('vy.csv')
by_tensor = torch.tensor(By_table.values, dtype=torch.float32,device=device)
vx_tensor = torch.tensor(vx_table.values, dtype=torch.float32,device=device)
rho_tensor = torch.tensor(rho_table.values, dtype=torch.float32,device=device)
vy_tensor = torch.tensor(vy_table.values, dtype=torch.float32,device=device)
p_tensor = torch.tensor(p_table.values, dtype=torch.float32,device=device)
vz_tensor=torch.tensor(np.zeros_like(E_tensor))
bz_tensor=torch.tensor(np.zeros_like(E_tensor))

def inter_condition(t,xr):
    matrix = torch.empty((len(xr)), 7)
    i=0
    for x in xr:
        p1,p2=np.floor([t * 6.756841139*(simt - 1), (x + 0.5) * (simx - 2)  + 1]).astype(int)
        matrix[i]=  ((        torch.as_tensor([  tn[p1,p2]  for tn in   [rho_tensor, vx_tensor,vy_tensor,vz_tensor,by_tensor,bz_tensor,p_tensor ]])))#rhovx_tensor,rhovy_tensor,rhovz_tensor,by_tensor,bz_tensor,E_tensor] ])   ) )
        i+=1
    return matrix

plot_model_predictions_at_time_point(model = loaded_model, inter_condition=inter_condition)