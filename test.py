import numpy as np
import cv2 
import torch
from torchdiffeq import odeint_adjoint as odeint
from model_nets import HDNet
import utils
import matplotlib.pyplot as plt

def run_traj(env, adj_net, hnet, hnet_decoder, env_name,
             num_trajs, time_steps, test_trained, phase2,
             save_video=False, video_path='videos/test.wmv'):
    
    if save_video:
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'WMV1')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (env.screen_width, env.screen_height), isColor=True)
        
    # Load models
    if test_trained:
        adj_net.load_state_dict(torch.load('models/' + env_name + '/adjoint.pth'))
        if phase2:
            hnet_decoder.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_decoder.pth'))
        else:
            hnet.load_state_dict(torch.load('models/' + env_name + '/hamiltonian_dynamics.pth'))

    # Build symplectic dynamics net from Hamiltonian net from phase 1 or Hamiltonian decoder from phase 2.
    if phase2:
        HDnet = HDNet(hnet=hnet_decoder)
    else:
        HDnet = HDNet(hnet=hnet)
    
    # Run optimal trajectory
    q = torch.tensor(env.sample_q(num_trajs, mode='test'), dtype=torch.float)
    # if save_video:
    #     q_0, _ = env.gym_env.reset()
    #     q = torch.tensor(np.array([q_0]), dtype=torch.float)
    p = adj_net(q)
    qp = torch.cat((q, p), axis=1)

    traj = odeint(HDnet, qp, torch.tensor(time_steps, requires_grad=False))
        
    print('Done finding trajectory...')
    print(traj[-1][-1])

    # Collect results and optionally save to videos
    final_costs_expected = []
    final_costs_controller = []
 

    qe, pe = torch.chunk(traj[-1], 2, dim=1)
    qe_np = qe.detach().numpy()
    pe_np = pe.detach().numpy()
    # pe = -adj_net(qe.to(torch.float32))

    qe_plot, pe_plot = torch.chunk(traj[:, 0, :], 2, dim=1)
    qe_plot_np = qe_plot.detach().numpy()
    pe_plot_np = pe_plot.detach().numpy()

    print(qe_plot.shape)

    # print("trajectory: ", traj[:, 0, :].shape)
    # print("trajectory: ", qe_plot_np)
    if save_video:

        fig, axes = plt.subplots(1, 3, figsize=(4*3, 4))

        axes[0].plot(qe_plot_np[:, 0], pe_plot_np[:, 0], marker='o')
        axes[0].set_xlabel('q1')
        axes[0].set_ylabel('p1')
        axes[0].set_title('Mountain Car in (q, p) space 1')

        axes[1].plot(qe_plot_np[:, 1], pe_plot_np[:, 1], marker='o')
        axes[1].set_xlabel('q2')
        axes[1].set_ylabel('p2')
        axes[1].set_title('Mountain Car in (q, p) space 2')

        # axes[0][2].plot(qe_plot_np[:, 2], pe_plot_np[:, 2], marker='o')
        # axes[0][2].set_xlabel('q3')
        # axes[0][2].set_ylabel('p3')
        # axes[0][2].set_title('Cart Pole in (q, p) space 3')

        axes[2].plot(qe_plot_np[:, 0], qe_plot_np[:, 1], marker='o')
        axes[2].set_xlabel('q')
        axes[2].set_ylabel('q\'')
        axes[2].set_title('Mountain Car in (q, q\') phase space')

        # axes[1][1].plot(qe_plot_np[:, 0], qe_plot_np[:, 1], marker='o')
        # axes[1][1].set_xlabel('position')
        # axes[1][1].set_ylabel('velocity')
        # axes[1][1].set_title('Cart Pole in horizontal space')

        # axes[1][2].plot(qe_plot_np[:, 2], qe_plot_np[:, 3], marker='o')
        # axes[1][2].set_xlabel('angle')
        # axes[1][2].set_ylabel('angular velocity')
        # axes[1][2].set_title('Cart Pole in angular space')

        plt.tight_layout()
        plt.show()


    qc = qe.clone()
    qc_np = qc.detach().numpy()
    pc = adj_net(qc.to(torch.float32))

    # print("State: ", q_np)
    for i in range(0, len(time_steps) - 1):
        final_costs_expected.append(env.eval(qe_np))
        final_costs_controller.append(env.eval(qe_np))
        control_coef = 0.5
        u_hat = (1.0/control_coef)*np.einsum('ijk,ij->ik', env.f_u(qc_np), -pc.detach().numpy())
        # u_hat = -pc.detach().numpy()
        # u_hat = torch.tensor([[1]])
        if save_video:
            # Write rendering image
            out.write(env.render(u_hat[0]))

        time_step = time_steps[i + 1] - time_steps[i]

        # if i < len(time_steps) - 2:
        #     qe, _ = torch.chunk(traj[i + 1], 2, dim=1)
        #     qe_np = qe.detach().numpy()
        #     pe = -adj_net(qe.to(torch.float32))

        #     qc = env.step(qc, torch.tensor(u_hat), dt=1)
        #     qc_np = qc.detach().numpy()
        #     pc = adj_net(qc.to(torch.float32))

    final_costs_expected.append(env.eval(qe_np))
    final_costs_controller.append(env.eval(qc_np))

    # Release video
    if save_video:
        out.release()

    env.close()
    # (num_traj, num_step)
    return np.swapaxes(np.array(final_costs_expected, dtype=float), 0, 1), np.swapaxes(np.array(final_costs_controller, dtype=float), 0, 1) 

def _test(env_name, num_trajs, time_steps, test_trained, phase2):
    # Initialize models (this first to take state dimension q_dim)
    _, adj_net, hnet, hnet_decoder, _, _ = \
        utils.get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    env = utils.get_environment(env_name) 

    return run_traj(env, adj_net, hnet, hnet_decoder, env_name,
                    num_trajs, time_steps,
                    test_trained, phase2)

def benchmarks(env_name, num_trajs, time_steps):
    # Calculate result
    final_costs_untrained, final_costs_untrained_c = _test(env_name, num_trajs, time_steps, test_trained=False, phase2=False)
    final_costs_phase_1, final_costs_phase_1_c = _test(env_name, num_trajs, time_steps, test_trained=True, phase2=False)
    final_costs_phase_2, final_costs_phase_2_c = _test(env_name, num_trajs, time_steps, test_trained=True, phase2=True)

    # Draw (statistical) plot
    eval_dict = {
        'Random Hamiltonian': final_costs_untrained,
        'Random Hamiltonian w/ control': final_costs_untrained_c,
        'NeuralPMP-phase 1': final_costs_phase_1,
        'NeuralPMP-phase 1 w/ control': final_costs_phase_1_c,
        'NeuralPMP': final_costs_phase_2,
        'NeuralPMP w/ control': final_costs_phase_2_c,
    }
    utils.plot_eval_benchmarks(eval_dict, time_steps,
                               title='Benchmarkings on ' + env_name,
                               plot_dir=env_name + '_benchmarks_plot.png')
    # Report statistics
    end_cost_untrained = np.mean(final_costs_untrained[:, -1])
    end_cost_phase_1 = np.mean(final_costs_phase_1[:, -1])
    end_cost_phase_2 = np.mean(final_costs_phase_2[:, -1])
    print('Random Hamiltonian:', end_cost_untrained)
    print('NeuralPMP-phase 1:', end_cost_phase_1)
    print('NeuralPMP:', end_cost_phase_2)

def visualize(env_name, time_steps, test_trained, phase2):
    # Initialize video path
    video_path = 'videos/test_'+ env_name +'.wmv'
    if not test_trained:
        print('\nTest untrained ' + env_name + ':')
        video_path = 'videos/test_'+ env_name +'_untrained.wmv'
    elif phase2:
        print('Test phase 2 for ' + env_name + ':')
        video_path = 'videos/test_'+ env_name +'_phase2.wmv'
    else:
        print('Test ' + env_name + ':')

    # Initialize/load models
    _, adj_net, hnet, hnet_decoder, _, _ = \
        utils.get_architectures(arch_file='models/architectures.csv', env_name=env_name)
    env = utils.get_environment(env_name) 
    run_traj(env, adj_net, hnet, hnet_decoder, env_name,
             1, time_steps, test_trained, phase2,
             save_video=True, video_path=video_path)
