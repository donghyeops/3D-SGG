from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

# 3D OM 참고 사이트
# https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids

if not __name__ == '__main__':
    '''
    fig_om = plt.figure()
    ax_om = fig_om.gca(projection='3d')
    ax_om.set_aspect('equal')
    '''
    fig = plt.figure(figsize=plt.figaspect(0.5))
    predict_ax = fig.add_subplot(1, 2, 1, projection='3d')
    predict_ax.set_aspect('equal')
    answer_ax = fig.add_subplot(1, 2, 2, projection='3d')
    answer_ax.set_aspect('equal')

def draw_object_3d_map(pos, labels, colors, draw_answer=False, output_path='om_map.jpg'):
    if draw_answer:
        ax = answer_ax
    else:
        ax = predict_ax
    #print('[draw] obj_len:', len(pos))
    positions = pos[:, :3].tolist()
    positions_convert = [[pos[0], pos[2], pos[1]] for pos in positions]
    sizes = pos[:, 3:].tolist()
    sizes_convert = [[size[0], size[2], size[1]] for size in sizes]
    colors = [colors[int(label)] for label in labels]

    ax.cla()
    ax.set_xlim([-2.5, 2.5])  # 양옆
    ax.set_ylim([-2, 3])  # 깊이
    ax.set_zlim([0, 5])  # 높이
    #print('[draw] pos:', pos)

    pc = plotCubeAt2(positions_convert, sizes_convert, colors=colors, edgecolor="k")
    ax.add_collection3d(pc)

    #plt.imsave('om_map.jpg')
    fig.savefig(fname=output_path, bbox_inches='tight', pad_inches=0)
    '''
    ax.set_xlim([-4,6])
    ax.set_ylim([4,13])
    ax.set_zlim([-3,9])
    '''
    return output_path

def draw_agent(pos, draw_answer=False, size=[(0.5,0.5,0.5)], output_path='om_map.jpg'):
    if draw_answer:
        ax = answer_ax
    else:
        ax = predict_ax

    positions_convert = [[pos[0], pos[2], pos[1]]]
    #print(positions_convert)
    colors = ['blue']

    pc = plotCubeAt2(positions_convert, size, colors=colors, alpha=1., edgecolor="k")
    ax.add_collection3d(pc)

    fig.savefig(fname=output_path, bbox_inches='tight', pad_inches=0)

    return output_path

def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, alpha=0.5, **kwargs):
    if not isinstance(colors,(list,np.ndarray)):
        colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)):
        sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    cube = Poly3DCollection(np.concatenate(g), alpha=alpha, **kwargs)
    face_color = np.repeat(colors,6)
    cube.set_facecolor(face_color)  # 이렇게 따로 설정해야 위 alpha이 적용됨 (버그인듯)
    return cube


if __name__ == '__main__':
    positions = [[0.5, 0.2, 0.9],[0.5, 0.2, 0.5]]
    sizes = [(0.2, 0.2, 0.2), (0.9, 0.5, 0.2)]
    colors = ["crimson","limegreen"]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    #fig, axes = plt.subplots(1, 2, projection='3d')
    #plt.subplots()
    axes = []
    axes.append(fig.add_subplot(1, 2, 1, projection='3d'))
    axes.append(fig.add_subplot(1, 2, 2, projection='3d'))
    #ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    pc = plotCubeAt2(positions,sizes,colors=colors, edgecolor="k")
    pc2 = plotCubeAt2(positions, sizes, alpha=1., colors=colors, edgecolor="k")
    #pc3 = plotCubeAt2(positions, sizes, alpha=0., colors=colors, edgecolor="k")
    axes[0].add_collection3d(pc)
    axes[1].add_collection3d(pc2)
    #axes[1].cla()

    '''
    ax.set_xlim([-4,6])
    ax.set_ylim([4,13])
    ax.set_zlim([-3,9])
    '''
    for i in range(2):
        axes[i].set_xlim([-1,1])
        axes[i].set_ylim([-1, 1])
        axes[i].set_zlim([-1, 1])
    #ax.set_xlim([-1,1])
    #ax.set_ylim([-1,1])
    #ax.set_zlim([-1,1])
    plt.show()