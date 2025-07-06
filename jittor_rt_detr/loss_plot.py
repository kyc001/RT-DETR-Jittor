import numpy as np
import matplotlib.pyplot as plt
loss = np.load('loss_curve.npy')
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')
plt.show()
