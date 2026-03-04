"""Compare solid vs porous simulation results."""
import numpy as np

# Load results
solid = np.load('results/solid/timeseries.npz', allow_pickle=True)
porous = np.load('results/porous/timeseries.npz', allow_pickle=True)

print('=== SOLID (final state) ===')
print(f'  p_max: {solid["p"][-1].max()/1e6:.2f} MPa')
print(f'  rho_max: {solid["rho"][-1].max():.2f} kg/m3')
print(f'  T_max: {solid["T"][-1].max():.1f} K')

print()
print('=== POROUS (final state) ===')
print(f'  p_max: {porous["p"][-1].max()/1e6:.2f} MPa')
print(f'  rho_max: {porous["rho"][-1].max():.2f} kg/m3')
print(f'  T_max: {porous["T"][-1].max():.1f} K')

print()
print('=== ANALYSIS ===')
print('velocity_offset = -119.2 m/s means gas velocity < piston velocity')
print('This means gas is leaving the domain through the piston')
print('Expected: POROUS should have LOWER p, rho, T than SOLID')
print()

p_solid = solid["p"][-1].max()/1e6
p_porous = porous["p"][-1].max()/1e6
print(f'Difference: p_porous - p_solid = {p_porous - p_solid:.2f} MPa')
if p_porous > p_solid:
    print('ERROR: Porous has HIGHER pressure - this is wrong!')
else:
    print('OK: Porous has lower pressure as expected')
