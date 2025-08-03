import matplotlib.pyplot as plt

def plot_compare(u_operator, u_target, save_folder, option='train'):
    u_operator = u_operator.detach().cpu().numpy()
    u_target = u_target.detach().cpu().numpy()
    error_absolute = (u_operator - u_target)

    # plot imshow of traj and u_target
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    im0 = axs[0].imshow(u_target, cmap = 'jet', extent=[0, 1, 0, 1])
    axs[0].set_title('Reference')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('x')
    # set the colorbar the same height as the image
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(u_operator, cmap = 'jet', extent=[0, 1, 0, 1])
    axs[1].set_title('NODE Operator')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('x')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    im2 = axs[2].imshow(error_absolute, cmap = 'jet', extent=[0, 1, 0, 1])
    axs[2].set_title('Error')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('x')
    fig.colorbar(im2, ax=axs[2], format='%.2e', fraction=0.046, pad=0.04)     # colorbar using scientific notation
    plt.savefig(save_folder+'/compare_'+option+'.png', bbox_inches='tight')
    plt.close()

def plot_compare_NS(u_operator, u_target, save_folder, N_x, ti):
    u_operator = u_operator.detach().cpu().numpy().reshape((N_x, N_x))
    u_target = u_target.detach().cpu().numpy().reshape((N_x, N_x))
    error_absolute = (u_operator - u_target)

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    im0 = axs[0].imshow(u_target, cmap = 'jet', extent=[0, 1, 0, 1])
    axs[0].set_title('Reference')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(u_operator, cmap = 'jet', extent=[0, 1, 0, 1])
    axs[1].set_title('NODE Operator')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    im2 = axs[2].imshow(error_absolute, cmap = 'jet', extent=[0, 1, 0, 1])
    axs[2].set_title('Error')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    fig.colorbar(im2, ax=axs[2], format='%.2e', fraction=0.046, pad=0.04)
    plt.savefig(save_folder+'/compare_t_'+format(ti, '.1f')+'.png', bbox_inches='tight')
    plt.close()