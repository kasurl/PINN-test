import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 判断是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        ).to(device)

    def forward(self, t, x):
        u = self.layer(torch.cat([t, x], dim=1))
        return u


def physics_loss(model, t, x, alpha=0.5):
    u = model(t, x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]
    f = (u_t - alpha * u_xx).pow(2).mean()
    return f


def boundary_loss(model, t_bc, x_left, x_right):
    u_left = model(t_bc, x_left)
    u_right = model(t_bc, x_right)
    loss_left = (u_left - g1(t_bc)).pow(2).mean()
    loss_right = (u_right - g2(t_bc)).pow(2).mean()
    return loss_left + loss_right


# 定义边界条件函数
def g1(t):
    return torch.zeros_like(t)


def g2(t):
    return torch.zeros_like(t)


def initial_loss(model, x_ic):
    t_0 = torch.zeros_like(x_ic).to(device)
    u_init = model(t_0, x_ic)
    u_exact = f(x_ic)
    return (u_init - u_exact).pow(2).mean()


def f(x):
    return torch.sin(np.pi * x)


def train(model, optimizer, num_epochs):
    losses = []
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()

        # 随机采样 t 和 x，并确保 requires_grad=True
        t = torch.rand(3000, 1).to(device)
        x = torch.rand(3000, 1).to(device) * 2 - 1  # x ∈ [-1, 1]
        t.requires_grad = True
        x.requires_grad = True

        # 物理损失
        f_loss = physics_loss(model, t, x)

        # 边界条件损失
        t_bc = torch.rand(500, 1).to(device)
        x_left = -torch.ones(500, 1).to(device)
        x_right = torch.ones(500, 1).to(device)
        bc_loss = boundary_loss(model, t_bc, x_left, x_right)

        # 初始条件损失
        x_ic = torch.rand(1000, 1).to(device) * 2 - 1
        ic_loss = initial_loss(model, x_ic)

        # 总损失
        loss = f_loss + bc_loss + ic_loss
        loss.backward()
        optimizer.step()

        # 记录损失
        losses.append(loss.item())

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return losses


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = train(model, optimizer, num_epochs=10000)


def plot_loss(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, color='blue', lw=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss Curve', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_loss(losses)


def plot_solution(model):
    x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
    t = torch.full((100, 1), 0.0).to(device)  # 在 t=0 时绘制解
    with torch.no_grad():
        u_pred = model(t, x).cpu().numpy()

    # 参考解 u(0,x) = sin(πx)
    u_exact = np.sin(np.pi * x.cpu().numpy())

    plt.figure(figsize=(8, 5))
    plt.plot(x.cpu().numpy(), u_pred, label='Predicted Solution', color='red', lw=2)
    plt.plot(x.cpu().numpy(), u_exact, label='Exact Solution (Initial)', color='blue', lw=2, linestyle='dashed')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('u(t=0, x)', fontsize=14)
    plt.title('Heat Conduction Equation Solution at t=0', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pinn1.png')
    plt.show()


plot_solution(model)


def plot_solution_3d(model):
    # 创建 (x, t) 网格
    x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
    t = torch.linspace(0, 1, 100).unsqueeze(1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    # 将 X 和 T 拉平，方便模型预测
    x_flat = X.reshape(-1, 1).to(device)
    t_flat = T.reshape(-1, 1).to(device)

    with torch.no_grad():
        u_pred = model(t_flat, x_flat).cpu().numpy().reshape(100, 100)

    # 绘制三维曲面图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.cpu().numpy(), T.cpu().numpy(), u_pred, cmap='viridis')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_zlabel('u(t, x)', fontsize=12)
    ax.set_title('Solution of Heat Conduction Equation on (x, t) Plane', fontsize=14)
    plt.savefig('pinn3.png')
    plt.show()


plot_solution_3d(model)  # 三维曲面图


def plot_solution_contour(model):
    # 创建 (x, t) 网格
    x = torch.linspace(-1, 1, 100).unsqueeze(1).to(device)
    t = torch.linspace(0, 1, 100).unsqueeze(1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    # 将 X 和 T 拉平，方便模型预测
    x_flat = X.reshape(-1, 1).to(device)
    t_flat = T.reshape(-1, 1).to(device)

    with torch.no_grad():
        u_pred = model(t_flat, x_flat).cpu().numpy().reshape(100, 100)

    # 绘制二维等高线图
    plt.figure(figsize=(8, 6))
    plt.contourf(X.cpu().numpy(), T.cpu().numpy(), u_pred, 100, cmap='viridis')
    plt.colorbar(label='u(t, x)')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title('Heat Conduction Equation Solution', fontsize=14)

    plt.tight_layout()
    plt.savefig('pinn2.png')
    plt.show()

# plot_solution_contour(model)
# ape(-1, 1).to(device)

    # with torch.no_grad():
    #     u_pred = model(t_flat, x_flat).cpu().numpy().reshape(100, 100)
    #
    # # 绘制二维等高线图
    # plt.figure(figsize=(8, 6))
    # plt.contourf(X.cpu().numpy(), T.cpu().numpy(), u_pred, 100, cmap='viridis')
    # plt.colorbar(label='u(t, x)')
    #
    # plt.xlabel('x', fontsize=12)
    # plt.ylabel('t', fontsize=12)
    # plt.title('Heat Conduction Equation Solution', fontsize=14)
    #
    # plt.tight_layout()
    # plt.savefig('pinn2.png')
    # plt.show()


plot_solution_contour(model)