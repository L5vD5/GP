import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from GPR import GPR
from utils.dataloader import load

train_X, train_y = load('../data/trainInterpolate.json')
test_X, test_y = load('../data/valExtrapolate.json')
train_X = train_X[:100, :]
train_y = train_y[:100, :]
train_y = train_y.reshape(-1, 27)

print('maximum train X {}'.format(np.max(train_X)))
print('maximum test X {}'.format(np.max(test_X)))
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.set_title('GT')

gpr = GPR()
gpr.fit(train_X, train_y)
pred_y = gpr.predict(test_X)

def update(frame, data, line):
  line[0].set_data(data[frame, :, 0], data[frame, :, 1])
  line[0].set_3d_properties(data[frame, :, 2])

  return line

def update2(frame, data2, line2):
  line2[0].set_data(data2[frame, :, 0], data2[frame, :, 1])
  line2[0].set_3d_properties(data2[frame, :, 2])

  return line2

pred_y = pred_y.reshape(-1, 9, 3)
print(np.max((test_y - pred_y), axis=0))
print(np.argmax((test_y - pred_y), axis=0))

pred_y += [0.1, 0, 0]

# line = ax.plot(test_y[0, :, 0], test_y[0, :, 1], test_y[0, :, 2], '.-')
# line2 = ax.plot(pred_y[0, :, 0], pred_y[0, :, 1], pred_y[0, :, 2], '.-', c='r')
line = ax.plot(test_y[2676, :, 0], test_y[2676, :, 1], test_y[2676, :, 2], '.-')
line2 = ax.plot(pred_y[2676, :, 0], pred_y[2676, :, 1], pred_y[2676, :, 2], '.-', c='r')

# ani = FuncAnimation(fig, update, fargs=[test_y, line], frames=range(len(test_y)), interval=500)
# ani2 = FuncAnimation(fig, update2, fargs=[pred_y, line2], frames=range(len(test_y)), interval=500)
plt.show()
