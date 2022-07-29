import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# ----------------------------------------------------------------------------------------------------------------------
def visualize_images(dataiter):

    img1, img2, flow, valid = dataiter.next()

    img1 = img1.cpu()
    img2 = img2.cpu()
    flow = flow.cpu()
    valid = valid.cpu()

    img1 = img1.numpy()[0,]
    img2 = img2.numpy()[0,]
    flow = flow.numpy()[0,]
    valid = valid.numpy()[0,]


    plt.subplot(1, 4, 1)
    print(np.transpose(img1, (1, 2, 0)))
    plt.imshow(np.transpose(img1, (1, 2, 0)).astype(int))
    plt.title('input 1')

    plt.subplot(1, 4, 2)
    plt.imshow(np.transpose(img2, (1, 2, 0)).astype(int))
    plt.title('input 2')

    plt.subplot(1, 4, 3)
    plt.imshow(flow[0,])
    plt.title('GT - x')

    plt.subplot(1, 4, 4)
    plt.imshow(flow[1,])
    plt.title('GT - y')

    plt.show()
# ----------------------------------------------------------------------------------------------------------------------
def save_inputs(in1, in2, flow, path, id, input1_name, input2_name):

    in1 = in1.detach()
    in1 = in1.cpu().numpy()[0,]
    in2 = in2.detach()
    in2 = in2.cpu().numpy()[0,]
    flow = flow.detach()
    flow = flow.cpu().numpy()[0,]

    norm_2 = np.power(flow[0,], 2) + np.power(flow[1,], 2)
    norm_2 = np.sqrt(norm_2)
    plt.figure(1)
    plt.imshow(norm_2)
    plt.savefig(os.path.join(path, input1_name))
    plt.close()
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(np.transpose(in1, (1, 2, 0)).astype(int))
    # plt.title(input1_name)
    # plt.axis('off')
    # plt.subplot(2, 2, 2)
    # plt.imshow(np.transpose(in2, (1, 2, 0)).astype(int))
    # plt.title(input2_name)
    # plt.axis('off')
    # plt.subplot(2, 2, 3)
    # plt.imshow(flow[1, ...].astype(int))
    # plt.title(input2_name)
    # plt.axis('off')
    # plt.subplot(2, 2, 4)
    # plt.imshow(flow[0, ...].astype(int))
    # plt.title(input2_name)
    # plt.axis('off')
    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.savefig(os.path.join(path, 'intputs_' + str(id) + '.png'), bbox_inches='tight', dpi=500)
    # plt.close()