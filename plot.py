import matplotlib.pyplot as plt
# Define the data
epochs = [1, 2, 3, 4, 5]
train_losses = [0.4673322956800461, 0.3602512192487717, 0.34395359358787536, 0.3345411633849144, 0.3281898899793625]
validation_losses = [0.3749387618272927, 0.3641416111569496, 0.35883747079190176, 0.35607969210406015, 0.35272559286302824]
# Plot the train and validation losses
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')
# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Losses over 5 Epochs')
plt.xticks(epochs)  # Set x ticks from 1 to 5
plt.legend()
# Show plot
plt.grid(True)
plt.show()