import torch 
import torch.nn as nn 
import time 
import os
import json
import pandas as pd
import torch.optim as optim 
import torchvision

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def lipconstant(D,x,y,device):
    # Calculate interpolation
    b = x.shape[0]
    alpha = torch.rand((b,1,1,1),device=device)

    interp = (alpha * y + (1 - alpha) * x) #.flatten(end_dim=1)
    interp.requires_grad_()

    # Calculate discriminator on interpolated examples
    Di = D(interp)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).to(device)
    gradients = torch.autograd.grad(outputs=Di, inputs=interp, grad_outputs=gradout,
                                    create_graph=True, retain_graph=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1))

    # Return gradient penalty
    return torch.mean(gradients_norm).to(device)



def gradient_penalty(D,x,y,device):
    # Calculate interpolation
    b = x.shape[0]
    alpha = torch.rand((b,1,1,1),device=device)

    interp = (alpha * y + (1 - alpha) * x) #.flatten(end_dim=1)
    interp.requires_grad_()

    # Calculate discriminator on interpolated examples
    Di = D(interp)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).to(device)
    gradients = torch.autograd.grad(outputs=Di, inputs=interp, grad_outputs=gradout,
                                    create_graph=True, retain_graph=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1))

    # Return gradient penalty
    return torch.mean((gradients_norm - 1)**2).to(device)



def train(D,G,train_loader,folder,device, batch_size=128,nz=100,num_epochs=100,gpw=0.1,lr=0.0002,log_every=10, save_every=50):
    # save parameters in json
    data_dict = {'batch size': batch_size,
                'nz':nz,
                'num_epochs': num_epochs,
                'gpw':gpw,
                'lr':lr}
    with open(f'{folder}/params.json', 'w') as jsonfile:
        json.dump(data_dict, jsonfile, indent=4)

    real = train_loader.__iter__().__next__().to(device)
    G.apply(weights_init);
    D.apply(weights_init);
    optimD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    zviz = torch.randn(batch_size,nz,1,1).to(device)
    lossesD = []
    lossesG = []

    t0 = time.time()
    for epoch in range(num_epochs):
        # For each batch in the train_loader
        for i, batch in enumerate(train_loader, 0):

            ############################
            # Draw Batches of real and fake images
            y = batch.to(device)
            z = torch.randn(batch_size,nz,1,1).to(device)
            x = G(z)
            
            ############################
            # Update D network
            optimD.zero_grad()
            x_gen = x.detach()
            Dloss = - torch.mean(D(y)) + torch.mean(D(x_gen)) #+ gpw*gradient_penalty(D,x,y,device) #WGAN GP loss 
            Dloss.backward()
            optimD.step()
            lossesD.append(-Dloss.item())


            ############################
            # Update G network
            optimG.zero_grad()
            z = torch.randn(batch_size,nz,1,1).to(device)
            x = G(z)
            Gloss = -D(x).mean() 
            Gloss.backward()
            optimG.step()
            lossesG.append(Gloss.item())


            # save losses in csv
            losses = {'generator': lossesG,
            'discriminator': lossesD}
            L = pd.DataFrame(losses)
            L.to_csv(f'{folder}/losses.csv', index=False)

            ############################

        # Display training stats and visualize
        if epoch % save_every == 0:
            faked =  G(zviz)
            print('[%d/%d][%d/%d][%.4f s]\tLoss_D: %.4f\tLoss_G: %.4f\tLip(D): %.4f'
                % (epoch+1, num_epochs, i, len(train_loader), time.time()-t0, -Dloss.item(), Gloss.item(),lipconstant(D,real,faked,device)))
            # save image
            with torch.no_grad():
                z = torch.randn(batch_size,nz,1,1).to(device)
                genimages = G(z)
                grid_img = torchvision.utils.make_grid(genimages.to('cpu'), nrow=16)
                torchvision.utils.save_image(grid_img, f'{folder}/epoch_{epoch}.png')

        #if epoch % save_every == 0:
            torch.save(G.state_dict(), f'{folder}/wgan_{epoch}.pt')


    torch.save(G.state_dict(), f'{folder}/wgan_{epoch}.pt')

    print('Total learning time = ',time.time()-t0)
                

def create_training_folder(name):
    # Define a base directory where you want to save the training results
    base_dir = 'training_results'
    # Create a new directory with a unique name
    new_folder = os.path.join(base_dir, 'training_' + str(len(os.listdir(base_dir)) + 1)+name)
    os.makedirs(new_folder, exist_ok=True)  # Create the directory if it doesn't exist
    return new_folder