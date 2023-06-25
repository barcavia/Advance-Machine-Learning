import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------- Hyperparameters ----------
INPUT_DIM = 3
HIDDEN_DIM = 256
OUTPUT_DIM = 2
NUM_SAMPLES = 3000
NUM_EPOCHS = 3000
LEARNING_RATE = 1e-3
T = 1000
DT = 1.0 / T

# ---------- Conditional Diffusion Model Parameters ----------
NUM_CLASSES = 9
EMBEDDING_DIM = 10


# Define the diffusion model architecture
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes=None, embedding_dim=0, is_conditional=False):
        """
        Initializes the DiffusionModel class.

        Args:
        - input_dim (int): Dimensionality of the input.
        - hidden_dim (int): Dimensionality of the hidden layer.
        - output_dim (int): Dimensionality of the output.
        - num_classes (int, optional): Number of classes for conditional embedding (default: None).
        - embedding_dim (int, optional): Dimensionality of the embedding (default: 0).
        - is_conditional (bool, optional): Whether to use conditional embedding (default: False).
        """
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.is_conditional = is_conditional
        if is_conditional:
            self.embedding = nn.Embedding(num_classes, embedding_dim=embedding_dim)

    def forward(self, xt, t, class_labels=None):
        """
        Performs forward pass through the DiffusionModel.

        Args:
        - xt (Any): Input tensor.
        - t (Any): Time step tensor.
        - class_labels (Any): Class labels tensor for conditional embedding (default: None).

        Returns:
        - out (Any): Output tensor.
        """
        # concatenating xt with time step t, and potentially with the class embedding, in case of conditional
        x = torch.cat([xt, t, self.embedding(class_labels)], dim=1) if self.is_conditional else torch.cat([xt, t],
                                                                                                          dim=1)
        out = self.leaky_relu(self.fc1(x))
        out = self.leaky_relu(self.fc2(out))
        out = self.fc3(out)
        return out


def scheduler(t):
    """
    Calculates the exponential noise scheduler value based on the given time step.

    Args:
    - t (Any): Time step.

    Returns:
    - result: Exponential scheduler value.
    """
    return np.exp((5 * (t - 1)))


def scheduler_derivative(t):
    """
    Calculates the derivative of the exponential noise scheduler value based on the given time step.

    Args:
    - t (Any): Time step.

    Returns:
    - result: Derivative of the exponential scheduler value.
    """
    return 5 * np.exp((5 * (t - 1)))


def sample_2D_points(num_points):
    """
    Generates random 2D points within the square range.

    Args:
    - num_points (int): Number of 2D points to generate.

    Returns:
    - points (torch.Tensor): Tensor containing the generated 2D points.
    """
    # Sampling points uniformly, inside the suitable square
    points = np.random.uniform(low=-1, high=1, size=(num_points, 2))
    # Converting the numpy array to a PyTorch tensor
    points = torch.tensor(points, dtype=torch.float32)
    return points


def visualize_forward_process(xt, dt):
    """
    Visualizes the forward process trajectory starting from the given initial point.

    Args:
    - xt (Any): Initial point of the trajectory.
    - dt (Any): Time step - 1./T

    Returns:
    - None
    """
    trajectory = []
    for t in np.arange(0, 1, dt):
        trajectory.append(xt.detach().numpy())
        xt, _ = forward_process(xt, scheduler(DT))
    trajectory = np.array(trajectory)
    plot_forward_process_trajectory(trajectory, title='Forward Process Trajectory')


def plot_forward_process_trajectory(trajectory, title):
    """
    Plots the trajectory of points.

    Args:
    - trajectory (Any): Array containing the trajectory points.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    # plotting the trajectory
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.arange(trajectory.shape[0]), cmap='coolwarm')
    plt.colorbar(label='Time t')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()


def estimate_denoised(z, epsilon, sigma_t):
    """
    Estimates the denoised value based on the noisy point, noise, and noise level.

    Args:
    - z (Any): Noisy point.
    - epsilon (Any): Noise sample.
    - sigma_t (Any): Noise level.

    Returns:
    - denoised (Any): Denoised value.
    """
    return z - epsilon * sigma_t


def forward_process(x, sigma):
    """
    Simulates the forward process by adding noise to the input, according to Eq. 4

    Args:
    - x (Any): Input point.
    - sigma (Any): Noise level.

    Returns:
    - x_next (Any): Next point after adding noise.
    - epsilon (Any): Noise sample.
    """
    # Sampling epsilon ~ N(0, I)
    epsilon = torch.randn_like(x)
    x_next = x + (epsilon * sigma)
    return x_next, epsilon


def plot_loss(losses):
    """
    Plots the loss function over training batches.

    Args:
    - losses (Any): List of loss values.

    Returns:
    - None
    """
    # Plot the loss function over training batches
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def DDIM_sampling(denoiser, dt, z, is_DDPM=False, is_conditional=False, class_labels=None):
    """
    Performs DDIM sampling using the given denoiser.

    Args:
    - denoiser (Any): Diffusion model.
    - dt (Any): Time step - 1./T
    - z (Any): Initial noised point.
    - is_DDPM (bool): Indicates if DDPM (Denoising Diffusion Probabilistic Model) is used.
    - is_conditional (bool): Indicates if the denoiser is conditional.
    - class_labels (Any): Class labels for conditional denoiser.

    Returns:
    - z (Any): Final denoised point.
    - trajectory (Any): List of points in the resulted denoised trajectory.
    """
    denoiser.eval()
    trajectory = [z]
    epsilon = torch.zeros_like(z)
    for t in torch.arange(1, 0, -dt):
        # in case of DDPM, we sample noise ~ N(0, I/100)
        if is_DDPM and t != 0:
            std_dev = torch.sqrt(torch.tensor(1 / 100))  # standard deviation for N(0, I/100)
            epsilon = std_dev * torch.randn_like(z)
        cur_t = torch.full((z.size(0), 1), t.item())
        sigma_t = scheduler(cur_t)
        sigma_t_der = scheduler_derivative(cur_t)
        # calculating the predicted noise
        predicted_noise = denoiser(z, cur_t, class_labels).detach() if is_conditional else denoiser(z, cur_t).detach()
        # the reverse sampling step
        x_hat0 = estimate_denoised(z, predicted_noise, sigma_t)
        score_z = (x_hat0 - z) / (sigma_t ** 2)
        # calculating the lambda parameter, in order to control the added noise, in case of DDPM
        lambda_param = torch.sqrt(torch.abs(sigma_t_der * sigma_t * dt))
        dz = -sigma_t_der * sigma_t * score_z * -dt
        z = z + dz + lambda_param * epsilon
        trajectory.append(z)
    return z, trajectory


def different_T_sampling(sampling_steps, denoiser, num_samples):
    """
    Perform different samplings of points using different numbers of sampling steps (T).
    Plot the sampled points on subplots.

    Args:
        sampling_steps (Any): List of sampling steps (T).
        denoiser: Denoiser model.
        num_samples (Any): Number of samples to generate.

    Returns:
        None
    """
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    for i, T in enumerate(sampling_steps):
        dt = 1.0 / T
        # sampling z ~ N(0, I)
        z = torch.randn(num_samples, 2)
        samples, _ = DDIM_sampling(denoiser, dt, z)
        row = i // 2
        col = i % 2
        axs[row, col].scatter(samples[:, 0], samples[:, 1], s=5)
        axs[row, col].set_title(f"T: {T}")

    plt.tight_layout()
    plt.show()


def perform_multiple_samplings(denoiser, dt, num_samples):
    """
    Perform multiple samplings of points using different seeds.
    Plot the sampled points on subplots.

    Args:
        denoiser: Denoiser model.
        dt (float): Time step - 1./T
        num_samples (int): Number of samples to generate.

    Returns:
        None
    """
    generate_seed = lambda: np.random.randint(0, num_samples)
    seeds = [generate_seed() for _ in range(9)]
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        # sampling z ~ N(0, I)
        z = torch.randn(num_samples, 2)
        samples, _ = DDIM_sampling(denoiser, dt, z)
        row = i // 3
        col = i % 3
        axs[row, col].scatter(samples[:, 0], samples[:, 1], s=5)
        axs[row, col].set_title(f"Seed: {seed}")

    plt.tight_layout()
    plt.show()


def plot_sigma_coefficients(sigma_coefficients_original, sigma_coefficients_modified):
    """
    Plot noise schedulers coefficients over time.

    Args:
        sigma_coefficients_original (Any): List of original noise scheduler coefficients.
        sigma_coefficients_modified (Any): List of modified noise scheduler coefficients.

    Returns:
        None
    """
    time = np.arange(len(sigma_coefficients_original))

    plt.figure(figsize=(10, 5))
    plt.plot(time, sigma_coefficients_original, label="Original Sampler")
    plt.plot(time, sigma_coefficients_modified, label="Modified Sampler")
    plt.title("Sigma Coefficients Over Time")
    plt.xlabel("Time")
    plt.ylabel("Sigma Coefficient")
    plt.legend()
    plt.grid(True)
    plt.show()


def sampling_using_same_noise(denoiser, dt):
    """
    Sample points using the same initial noise multiple times and visualize the results.

    Args:
        denoiser (Any): Denoising model.
        dt (float): Time step - 1./T

    Returns:
        None
    """
    # sampling z ~ N(0, I)
    # sampling a point from the model using the same initial noise 10 times
    z = torch.randn(1, 2)
    ddpm_sampled_points = []
    ddim_sampled_points = []
    trajectories = []
    for i in range(10):
        # denoising process after modification - results in different end points
        ddpm_samples, cur_trajectory = DDIM_sampling(denoiser, dt, z, is_DDPM=True)
        ddpm_sampled_points.append(ddpm_samples.detach())
        # denoising process before modification - results in the same end point
        ddim_samples, _ = DDIM_sampling(denoiser, dt, z, is_DDPM=False)
        ddim_sampled_points.append(ddim_samples.detach())
        # denoising trajectories of the first 4 points
        if i < 4:
            trajectories.append(torch.cat(cur_trajectory, dim=0).numpy())

    ddpm_sampled_points = torch.cat(ddpm_sampled_points, dim=0).numpy()
    ddim_sampled_points = torch.cat(ddim_sampled_points, dim=0).numpy()
    print("Before Sampling Modification: 10 points using the same initial noise input:")
    print(ddim_sampled_points)
    print("After Sampling Modification: 10 points using the same initial noise input:")
    print(ddpm_sampled_points)
    plot_denoising_trajectory_same_initial_noise(trajectories)


def plot_denoising_trajectory_same_initial_noise(trajectories):
    """
    Plot denoising trajectories using the same initial noise.

    Args:
        trajectories (Any): List of denoising trajectories.

    Returns:
        None
    """
    plt.figure()
    colors = ['purple', 'cyan', 'magenta', 'gray']
    for k, trajectory in enumerate(trajectories):
        plt.scatter(trajectory[:-1, 0], trajectory[:-1, 1], color=colors[k], label=f'Trajectory {k + 1}')
        last_point = trajectory[-1]
        # bolding the last point in the trajectory, with '*' marker
        plt.scatter(last_point[0], last_point[1], color=colors[k], edgecolors='black', marker='*', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    # Draw the outline of the [-1,1] x [-1,1] square
    plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], 'k-', linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.show()


def train_diffusion_model(model, points, optimizer, num_epochs, is_conditional=False, class_labels=None):
    """
    Train the diffusion model.

    Args:
        model (Any): Diffusion model.
        points (Any): Input points for training.
        optimizer (Any): Optimizer for training.
        num_epochs (int): Number of training epochs.
        is_conditional (bool): Whether the model is conditional or unconditional.
        class_labels (Any): Class labels for conditional model.

    Returns:
        list: List of training losses.
    """
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # generating different t for each sample
        t = torch.rand(points.size(0), 1)
        sigma = scheduler(t)
        noised_points, epsilon = forward_process(points, sigma)
        # calculating the predicted noise
        predicted_noise = model(noised_points, t, class_labels) if is_conditional else model(noised_points, t)
        loss = criterion(predicted_noise, epsilon)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    return losses


def unconditional_diffusion_model(points):
    """
    Train an unconditional diffusion model and perform various analyses.

    Args:
        points (torch.Tensor): Input points for training.

    Returns:
        None
    """
    # # setting random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

    # question 1 - the forward process of a point outside the square as a trajectory in a 2D space
    # choose a starting point inside the square
    x0 = torch.tensor([0.0, 0.0])
    visualize_forward_process(x0, DT)
    model = DiffusionModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = train_diffusion_model(model, points, optimizer, NUM_EPOCHS)
    # question 2 - present the loss function over the training batches of the denoiser
    plot_loss(losses)
    # question 3 - different samplings of 1000 points, using 9 different seeds
    perform_multiple_samplings(model, DT, num_samples=1000)
    # question 4 - sampling results for different numbers of sampling steps, T
    sampling_steps = [1, 10, 100, 1000]
    different_T_sampling(sampling_steps, model, num_samples=1000)
    # question 5 - modifying the given sampler and plotting the σ coefficients over time.
    orig_sigma_coeffs = [scheduler(t) for t in np.arange(0, 1, DT)]
    # modifying the sigma coefficients by applying a scaling factor
    modified_scheduler = lambda x: np.sqrt(x)
    modified_sigma_coeffs = [modified_scheduler(t) for t in np.arange(0, 1, DT)]
    plot_sigma_coefficients(orig_sigma_coeffs, modified_sigma_coeffs)
    # question 6 - sampling using the same noise
    sampling_using_same_noise(model, DT)


def plot_points(points, class_labels, title, colors):
    """

    Plot the given points with class labels.

    Args:
        points (Any): Array of points.
        class_labels (Any): Array of class labels.
        title (str): Title of the plot.
        colors (list): List of colors for different classes.

    Returns:
        None
    """
    cmap = ListedColormap(colors)
    plt.scatter(points[:, 0], points[:, 1], c=class_labels, cmap=cmap)
    plt.colorbar(ticks=range(len(np.unique(class_labels))), label="Class")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.show()


def sample_classes_points(num_samples, samples_labels, conditional_model, colors):
    """
    Sample points from the conditional diffusion model and plot them with class labels.

    Args:
        num_samples (int): Number of points to sample.
        samples_labels (Any): Array of class labels for the sampled points.
        conditional_model: The conditional diffusion model.
        colors (list): List of colors for different classes.

    Returns:
        tuple: A tuple containing the sampled points and the corresponding trajectories.
    """
    initial_noise = torch.randn(num_samples, 2)
    samples, traj = DDIM_sampling(conditional_model, DT, initial_noise,
                                  is_conditional=True, class_labels=samples_labels)
    plot_points(samples, samples_labels, str(num_samples) + ' Sampled Points (Colored by Classes)', colors)
    return samples, traj


def get_class_labels(points):
    """
    Get the class labels for the given points.

    Args:
        points (Any): Array of points.

    Returns:
        Array of class labels.
    """
    subspace_size = 2 / 3
    class_labels = np.floor((points[:, 0] + 1) / subspace_size) + np.floor((points[:, 1] + 1) / subspace_size) * 3
    return class_labels.long()


def estimate_probability(x, model, class_labels, dt):
    """
    Estimate the probability using the diffusion model.

    Args:
        x (Any): Input tensor.
        model: The diffusion model.
        class_labels: Class labels for the input tensor.
        dt: Time step value.

    Returns:
        Estimated probability.
    """
    SNR = lambda sigma_t: 1.0 / (sigma_t ** 2)
    # number of noise and time combinations
    num_combinations = 2000
    SNR_diff_sum = 0.0
    for _ in range(num_combinations):
        t = torch.rand(x.size(0), 1)
        sigma_t = scheduler(t)
        xt, epsilon = forward_process(x, sigma_t)
        predicted_noise = model(xt, t, class_labels)
        # computing SNR difference for the sample t value
        norm_2 = torch.norm(x - estimate_denoised(xt, predicted_noise, sigma_t)) ** 2
        SNR_diff = (SNR(scheduler(t - dt)) - SNR(sigma_t)) * norm_2
        SNR_diff_sum += SNR_diff.item()

    # Average the SNR difference
    elbo = 0.5 * T * (SNR_diff_sum / num_combinations)

    return -elbo


def plot_denoising_trajectories(colors, samples_labels, trajectory):
    """
    Plot denoising trajectories for each sample label.

    Args:
        colors (Any): A list of colors corresponding to each sample label.
        samples_labels (Any): A list of sample labels.
        trajectory (Any): Denoising trajectories.

    Returns:
        None
    """
    plt.figure()
    # plotting each class sample trajectory
    for label in samples_labels:
        cur_label_trajectory = trajectory[label]
        plt.scatter(cur_label_trajectory[:-1, 0], cur_label_trajectory[:-1, 1], label=f'Class {label}',
                    color=colors[label])
        last_point = cur_label_trajectory[-1]
        # bolding the last point in the trajectory, with '*' marker
        plt.scatter(last_point[0], last_point[1], color=colors[label], edgecolors='black', marker='*', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    # Draw the outline of the [-1,1] x [-1,1] square
    plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], 'k-', linewidth=2)
    # Set tick positions and labels for x-axis
    x_ticks = np.arange(-1, 1.25, 0.25)
    plt.xticks(x_ticks)
    # Set tick positions and labels for y-axis
    y_ticks = np.arange(-1, 1.25, 0.25)
    plt.yticks(y_ticks)
    plt.grid(True)
    plt.legend()
    plt.show()


def train_and_sample_diffusion_model(points):
    """
    Training a conditional diffusion model, and sampling 2000 points

    Args:
        points (Any): Input points for training.

    Returns:
        the samples achieved using the model
    """
    # calculation the class labels for each point
    input_class_labels = get_class_labels(points)
    # defining the diffusion model architecture
    conditional_model = DiffusionModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                                       num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, is_conditional=True)
    optimizer = optim.Adam(conditional_model.parameters(), lr=LEARNING_RATE)
    losses = train_diffusion_model(conditional_model, points, optimizer, NUM_EPOCHS, is_conditional=True,
                                   class_labels=input_class_labels)

    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'gray']
    # sampling 2000 points, plotting and coloring them by their classes
    samples, _ = sample_classes_points(num_samples=2000,
                                       samples_labels=torch.randint(low=0, high=NUM_CLASSES, size=(2000,)),
                                       conditional_model=conditional_model, colors=colors)

    return samples


def conditional_diffusion_model(points):
    """
    Train an conditional diffusion model and perform various analyses.

    Args:
        points (Any): Input points for training.

    Returns:
        None
    """
    # calculation the class labels for each point
    input_class_labels = get_class_labels(points)
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'gray']
    # question 1 - plotting the points coloring by their classes.
    plot_points(points, input_class_labels, title='Input Points (Colored by Classes)', colors=colors)
    conditional_model = DiffusionModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                                       num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM, is_conditional=True)
    optimizer = optim.Adam(conditional_model.parameters(), lr=LEARNING_RATE)
    losses = train_diffusion_model(conditional_model, points, optimizer, NUM_EPOCHS, is_conditional=True,
                                   class_labels=input_class_labels)
    plot_loss(losses)

    # question 3 - sampling 1 point from each class
    samples_labels = torch.arange(0, NUM_CLASSES)
    samples, trajectory = sample_classes_points(num_samples=NUM_CLASSES, samples_labels=samples_labels,
                                                conditional_model=conditional_model, colors=colors)
    trajectory = np.transpose(torch.stack(trajectory).numpy(), (1, 0, 2))
    plot_denoising_trajectories(colors, samples_labels, trajectory)
    # question 4 - sampling 2000 points using the conditional model
    sample_classes_points(num_samples=2000, samples_labels=torch.randint(low=0, high=NUM_CLASSES, size=(2000,)),
                          conditional_model=conditional_model, colors=colors)

    # question 6 - estimating points probability
    # choosing the points to be estimated
    points_to_be_estimated = [[torch.tensor([[0.5, 0.5]]), " defined within the input distribution, true class"],
                              [torch.tensor([[2.0, 2.0]]), " defined outside the input distribution"],
                              [torch.tensor([[0.75, 0.0]]),
                               " defined in the same location as the next point, true class"],
                              [torch.tensor([[0.75, 0.0]]),
                               " defined in the same location as the previous point, wrong class"],
                              [torch.tensor([[-0.5, -0.5]]), " defined within the input distribution, true class"],
                              [torch.tensor([[-0.9, 0.9]]), " defined within the input distribution, wrong class"]]
    # setting the points labels according to the instructions
    points_labels = [torch.tensor([8]),
                     torch.tensor([0]),
                     torch.tensor([5]),
                     torch.tensor([4]),
                     torch.tensor([0]),
                     torch.tensor([2])]
    for i, x in enumerate(points_to_be_estimated):
        print("The point estimation of: " + str(x[0].squeeze().tolist()) + x[1] + " is: " + str(
            estimate_probability(x[0], conditional_model, points_labels[i], DT)))


if __name__ == '__main__':
    # sampling the training data for the diffusion process
    points = sample_2D_points(NUM_SAMPLES)

    # in order to activate the results for the PDF questions, remove the following comments:
    # --------------------------------------
    #    Part One - Unconditional Model
    # --------------------------------------
    # unconditional_diffusion_model(points)

    # --------------------------------------
    #    Part Two - Conditional Model
    # --------------------------------------
    # conditional_diffusion_model(points)

    # training conditional diffusion model and sampling points, according to the submission instructions:
    samples = train_and_sample_diffusion_model(points)
