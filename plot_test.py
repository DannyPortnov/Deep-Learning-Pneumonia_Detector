
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


x = [i for i in range(11)]
# y = [i for i in range(11)]
# Create a figure and subplot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot the training and validation accuracy
axs[0].plot(x, 'bo-', label='Training Accuracy')
axs[0].plot(x, 'ro-', label='Validation Accuracy')
axs[0].set_title(f'Training & Validation Accuracy\n opt=RMSPROPDFS, Momentum=0.9, lr={0.0000001}, Epochs={40}')
# axs[0].title.set_size(10) # if title is too big, change the size here 
axs[0].legend(loc='lower right')
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Accuracy")

# Plot the training and validation loss
axs[1].plot(x, 'bo-', label='Training Loss')
axs[1].plot(x, 'ro-', label='Validation Loss')
axs[1].set_title(f'Training & Validation Loss\n opt=RMSPROPDFS, Momentum=0.9, lr={0.0000001}, Epochs={40}')
# axs[1].title.set_size(10) # if title is too big, change the size here 
axs[1].legend(loc='upper right')
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss")

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.2)

plt.show()
