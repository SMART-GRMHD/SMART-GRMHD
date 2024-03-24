import torch
import numpy as np

# model parameter
Gamma = 5 / 3
B_x = 5


def m_x(D, h, gamma, B_x, B_y, B_z, v_x, v_y, v_z):
    return (D * h * gamma + (B_x**2 + B_y**2 + B_z**2)) * v_x - (
        v_x * B_x + v_y * B_y + v_z * B_z
    ) * B_x


def m_y(D, h, gamma, B_x, B_y, B_z, v_x, v_y, v_z):
    return (D * h * gamma + (B_x**2 + B_y**2 + B_z**2)) * v_y - (
        v_x * B_x + v_y * B_y + v_z * B_z
    ) * B_y


def m_z(D, h, gamma, B_x, B_y, B_z, v_x, v_y, v_z):
    return (D * h * gamma + (B_x**2 + B_y**2 + B_z**2)) * v_z - (
        v_x * B_x + v_y * B_y + v_z * B_z
    ) * B_z


def E(D, h, gamma, B_x, B_y, B_z, v_x, v_y, v_z, p):
    return (
        D * h * gamma
        - p
        + (B_x**2 + B_y**2 + B_z**2) / 2
        + (
            (v_x**2 + v_y**2 + v_z**2) * (B_x**2 + B_y**2 + B_z**2)
            - (v_x * B_x + v_y * B_y + v_z * B_z) ** 2
        )
        / 2
    )


def D(rho, gamma):
    return rho * gamma


def h(Gamma, p, rho):
    return 1 + ((Gamma) / (Gamma - 1)) * (p / rho)


def gamma(v):
    speed_squared = torch.sum(v**2)
    if speed_squared >= 1:
        # Handle edge case or throw an error
        # For example, you can clip the value to just below 1
        speed_squared = 0.999999
    return 1 / np.sqrt(1 - speed_squared)


def momentum(D, h, gamma, B, v):
    magnetic_field_term = torch.sum(B**2)
    return (D * h * gamma + magnetic_field_term) * v - torch.dot(v, B) * B


def energy(D, h, gamma, B, v, p):
    kinetic_energy_term = torch.sum(v**2)
    magnetic_energy_term = torch.sum(B**2)
    return (
        D * h * gamma
        - p
        + magnetic_energy_term / 2
        + 0.5 * (kinetic_energy_term * magnetic_energy_term - torch.dot(v, B) ** 2)
    )

def conserved_alfredo(P, B_x=B_x, Gamma=Gamma):
    t_P = P.t().squeeze()  # .to(device)
    rho = t_P[0]
    v_x = t_P[1]
    v_y = t_P[2]
    v_z = t_P[3]
    B_y = t_P[4]
    B_z = t_P[5]
    p = t_P[6]

    B2 = B_x**2 + B_y**2 + B_z**2
    v2 = v_x**2 + v_y**2 + v_z**2
    vB = v_x * B_x + v_y * B_y + v_z * B_z

    hh = 1 + p * Gamma / (rho * (Gamma - 1))
    gg = (1 - v2) ** (-0.5)
    DD = rho * gg

    mx = (DD * hh * gg + B2) * v_x - vB * B_x
    my = (DD * hh * gg + B2) * v_y - vB * B_y
    mz = (DD * hh * gg + B2) * v_z - vB * B_z

    EE = DD * hh * gg - p + B2 / 2 + (v2 * B2 - vB**2) / 2

    return torch.stack([DD, mx, my, mz, B_y, B_z, EE]).t()


def currents_alfredo(P, B_x=B_x, Gamma=Gamma):
    t_P = P.t().squeeze()  # .to(device)
    rho = t_P[0]
    v_x = t_P[1]
    v_y = t_P[2]
    v_z = t_P[3]
    B_y = t_P[4]
    B_z = t_P[5]
    p = t_P[6]


    B2 = B_x**2 + B_y**2 + B_z**2
    v2 = v_x**2 + v_y**2 + v_z**2
    vB = v_x * B_x + v_y * B_y + v_z * B_z

    hh = 1 + p * Gamma / (rho * (Gamma - 1))
    gg = (1 - v2) ** (-0.5)
    DD = rho * gg

    beta2 = B2 / (gg * gg) + vB**2

    wtt = rho * hh + p - beta2
    bx = B_x / gg + gg * v_x * vB
    by = B_y / gg + gg * v_y * vB
    bz = B_z / gg + gg * v_z * vB
    pt = p + beta2 / 2

    mx = (DD * hh * gg + B2) * v_x - vB * B_x


    #EE = DD * hh * gg - p + B2 / 2 + (v2 * B2 - vB**2) / 2

    return torch.stack(
        [
            DD * v_x,
            wtt * gg * (v_x**2) - bx**2 + pt,
            wtt * gg * (v_x * v_y) - bx * by,
            wtt * gg * (v_x * v_z) - bx * bz,
            v_y * B_x - v_x * B_y,
            v_z * B_x - v_x * B_z,
            mx,
        ]
    ).t()