import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data>
csv_path = 'data.csv'
data_df = pd.read_csv(csv_path, sep=';')
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=31)

# set up data loading for the training and validation>
train_loader = t.utils.data.DataLoader(ChallengeDataset(train_df, mode='train'), batch_size=64, shuffle=True)
val_loader = t.utils.data.DataLoader(ChallengeDataset(val_df, mode='val'), batch_size=64)


# an instance of our ResNet model>
resnet_model = model.ResNet()


# the suitable loss criterion>
crit = t.nn.MSELoss()

# the optimizer>
optimizer = t.optim.Adam(resnet_model.parameters(), lr=1e-4, weight_decay=1e-5)
# optimizer = t.optim.SGD(model.parameters(), lr=1e-2, momentum=0.8)
# scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
# Optional learning rate scheduler
scheduler = None  # Uncomment the following line to use StepLR
# scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

# an object of type Trainer>
trainer = Trainer(
    model=resnet_model,
    crit=loss_fn,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=True,
    scheduler=scheduler
)

# fit the trainer>
res = trainer.fit(epochs=50)

# plot the results>
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')