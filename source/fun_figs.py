# %%
import navlie as nav
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# %% Banana distribution plot
def banana_plot(ax=None):
    N = 500
    x0 = nav.lib.SE2State([0.3, 3, 4], direction="right")
    covariance = np.diag([0.2**2, 0.05**2, 0.05**2])
    process_model = nav.lib.BodyFrameVelocity(np.zeros(3))

    dx_samples = nav.randvec(covariance, N).T
    x0_samples = [x0.plus(dx) for dx in dx_samples]

    # Monte-carlo the trajectory forward in time
    dt = 0.1
    T = 10
    stamps = np.arange(0, T, dt)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    final_states = []
    for sample in x0_samples:
        x_traj = [sample.copy()]
        u = nav.lib.VectorInput([0.1, 0.3, 0])
        x = sample
        for _ in stamps:
            x = process_model.evaluate(x, u, dt)
            x_traj.append(x.copy())

        # plot the trajectory
        traj_pos = np.array([x.position for x in x_traj])

        # random greyscale color
        color = np.random.uniform(0.3, 0.9)
        ax.plot(
            traj_pos[:, 0],
            traj_pos[:, 1],
            color=(color, color, color),
            zorder=1,
        )

        # save the final state
        final_states.append(x_traj[-1])

    final_positions = np.array([x.position for x in final_states])
    ax.scatter(
        final_positions[:, 0], final_positions[:, 1], color="C0", zorder=2
    )

    # Propagate the mean with EKF
    kf = nav.ExtendedKalmanFilter(process_model)
    x0_hat = nav.StateWithCovariance(x0, covariance)

    x_hat_traj = [x0_hat]
    for t in stamps:
        u.stamp = t
        x_hat_traj.append(kf.predict(x_hat_traj[-1], u, dt))

    mean_traj = np.array([x.state.position for x in x_hat_traj])
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], color="r", zorder=3, linewidth=3)
    ax.set_aspect("equal")


# banana_plot()


# %%
def pose3d_plot(ax=None):
    N = 500
    x0 = nav.lib.SE3State([0.3, 3, 4, 0, 0, 0], direction="right")
    process_model = nav.lib.BodyFrameVelocity(np.zeros(6))

    dt = 0.1
    T = 20
    stamps = np.arange(0, T, dt)

    x_traj = [x0.copy()]
    u = nav.lib.VectorInput([0.1, 0.3, 0, 1, 0, 0])
    x = x0.copy()
    for _ in stamps:
        x = process_model.evaluate(x, u, dt)
        x_traj.append(x.copy())

    fig, ax = nav.plot_poses(x_traj, ax=ax)


# pose3d_plot()


# %%
def three_sigma_plot(axs=None):
    dataset = nav.lib.datasets.SimulatedPoseRangingDataset()

    estimates = nav.run_filter(
        nav.ExtendedKalmanFilter(dataset.process_model),
        dataset.get_ground_truth()[0],
        np.diag([0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2]),
        dataset.get_input_data(),
        dataset.get_measurement_data(),
    )

    results = nav.GaussianResultList.from_estimates(
        estimates, dataset.get_ground_truth()
    )

    fig, axs = nav.plot_error(results[:, :3], axs=axs)
    axs[2].set_xlabel("Time (s)")


# three_sigma_plot()


if __name__ == "__main__":
    # Make one large figure which has all the plots. This will be a 1x3 grid, with the
    # last plot itself being a three vertically stacked plots.

    # The following values where chosen by trial and error
    # top=0.975,
    # bottom=0.097,
    # left=0.025,
    # right=0.992,
    # hspace=0.2,
    # wspace=0.117

    # which will be used here:

    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], projection="3d")

    # The last plot is a 3x1 grid
    gs2 = gs[2].subgridspec(3, 1, hspace=0.1)
    ax3 = fig.add_subplot(gs2[0])
    ax4 = fig.add_subplot(gs2[1])
    ax5 = fig.add_subplot(gs2[2])

    # Remove tick labels for ax3 and ax4
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])

    # Remove all tick labels for ax2
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    banana_plot(ax1)
    pose3d_plot(ax2)
    three_sigma_plot(np.array([ax3, ax4, ax5]))

    # Set spacing to the above values
    fig.subplots_adjust(
        top=0.975,
        bottom=0.097,
        left=0.025,
        right=0.992,
        hspace=0.2,
        wspace=0.117,
    )

    # Save the figure with transparent background, next to this file
    import os

    fig.savefig(
        os.path.join(os.path.dirname(__file__), "fun_figs.png"),
        transparent=True,
    )

    plt.show()
# %%
