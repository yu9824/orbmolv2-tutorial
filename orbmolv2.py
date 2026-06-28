# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: orbmolv2_312
#     language: python
#     name: python3
# ---

# %%
import io
from array import array

import matplotlib.pyplot as plt
from ase import Atoms
from ase.optimize import FIRE
from ase.visualize import view
from orb_models.forcefield.inference.calculator import ORBCalculator
from orb_models.forcefield.pretrained import orbmol_v2
from tqdm.auto import tqdm

# %%
atoms_ammonium = Atoms(
    "NH4",
    positions=[
        (0.000, 0.000, 0.000),  # N
        (0.629, 0.629, 0.629),
        (-0.629, -0.629, 0.629),
        (-0.629, 0.629, -0.629),
        (0.629, -0.629, -0.629),
    ],
)
atoms_ammonium.info["charge"] = 1
atoms_ammonium.info["spin"] = 1

view(atoms_ammonium, viewer="ngl")

# %%
device = "cuda"

model, atoms_adapter = orbmol_v2(
    "./checkpoints/orbmol-v2-teqabfhg-20260523.ckpt", device=device
)

# %%
calc = ORBCalculator(model, atoms_adapter=atoms_adapter, device=device)
atoms_ammonium.calc = calc


# %%
atoms_ammonium.get_potential_energy()

# %% [markdown]
# 以下の条件で計算
#
# > 分子の配置は分子間の距離がそれぞれ10 nmとなるように直線状に配置した。構造最適化はASEライブラリのFIRE法を使い、力の収束条件を fmax=0.01 (eV/Å, 1 Å=0.1 nm) とした。
# > 
# > <cite>https://tech.preferred.jp/ja/blog/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%9D%E3%83%86%E3%83%B3%E3%82%B7%E3%83%A3%E3%83%AB%E3%81%AE%E8%BF%91%E4%BC%BC%E3%81%AE%E9%99%90%E7%95%8C%E3%82%92%E8%B6%85%E3%81%88%E3%81%A6%E3%80%90%E3%82%A4/</cite>

# %%
spacing = 100.0  # 10 nm
force_threshold = 0.01

n_mol_list = range(1, 11)

energies_charge_state_orbmolv2 = array("f")
energies_spin_state_orbmolv2 = array("f")
n_molecules_array = array("I", n_mol_list)

for n_molecules in tqdm(n_mol_list):
    print(f"{n_molecules=}")

    system = Atoms()

    # build linear chain
    for i in range(n_molecules):
        mol = atoms_ammonium.copy()
        mol.translate([i * spacing])
        system += mol

    assert len(system) == n_molecules * len(atoms_ammonium)

    # -------------------------
    # charged state calculation
    # -------------------------
    system_charge = system.copy()
    system_charge.info["charge"] = n_molecules
    system_charge.info["spin"] = 1
    system_charge.calc = calc

    opt = FIRE(system_charge)
    opt.run(fmax=force_threshold)

    energies_charge_state_orbmolv2.append(system_charge.get_potential_energy())

    # -------------------------
    # spin state calculation
    # -------------------------
    system_spin = system.copy()
    system_spin.info["charge"] = 0
    system_spin.info["spin"] = n_molecules + 1
    system_spin.calc = calc

    opt = FIRE(system_spin)
    opt.run(fmax=force_threshold)

    energies_spin_state_orbmolv2.append(system_spin.get_potential_energy())

energies_charge_per_mol_orbmolv2 = array(
    "f",
    [e / n for n, e in zip(n_mol_list, energies_charge_state_orbmolv2)],
)

energies_spin_per_mol_orbmolv2 = array(
    "f",
    [e / n for n, e in zip(n_mol_list, energies_spin_state_orbmolv2)],
)

# %%
# reference: https://tech.preferred.jp/ja/blog/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%9D%E3%83%86%E3%83%B3%E3%82%B7%E3%83%A3%E3%83%AB%E3%81%AE%E8%BF%91%E4%BC%BC%E3%81%AE%E9%99%90%E7%95%8C%E3%82%92%E8%B6%85%E3%81%88%E3%81%A6%E3%80%90%E3%82%A4/
energies_charge_per_mol_uma1p2 = array(
    "f",
    [
        -1547.90,
        -1547.20,
        -1546.82,
        -1546.58,
        -1545.63,
        -1544.58,
        -1543.73,
        -1542.76,
        -1541.87,
        -1541.02,
    ],
)
energies_spin_per_mol_uma1p2 = array(
    "f",
    [
        -1552.17,
        -1550.30,
        -1551.54,
        -1551.78,
        -1551.99,
        -1552.03,
        -1552.08,
        -1551.96,
        -1551.80,
        -1551.77,
    ],
)


# %%
fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=144)
ax.plot(
    n_molecules_array,
    energies_charge_per_mol_uma1p2,
    label="Multiple NH4+ (charge=N,spin=1, UMA 1.2)",
    marker="o",
    color="#6a3e97",
)
ax.plot(
    n_molecules_array,
    energies_charge_per_mol_orbmolv2,
    label="Multiple NH4+ (charge=N,spin=1, OrbMol v2)",
    marker="^",
)
ax.plot(
    n_molecules_array,
    energies_spin_per_mol_uma1p2,
    label="Multiple NH4 (charge=0, spin=N+1, UMA 1.2)",
    marker="s",
    color="#349d79",
)
ax.plot(
    n_molecules_array,
    energies_spin_per_mol_orbmolv2,
    label="Multiple NH4 (charge=0, spin=N+1, OrbMol v2)",
    marker="v",
)
ax.set_xlabel("Number of molecules")
ax.set_ylabel("Energy per molecule / eV")
ax.legend()
fig.tight_layout()
