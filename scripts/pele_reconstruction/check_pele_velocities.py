"""Check PeleC velocity data to understand offset."""
from pathlib import Path
from pele_data_loader import PeleDataLoader

data_dir = Path('pele_data/truncated_raw_data')
loader = PeleDataLoader(data_dir)
data = loader.load()

print('Time range:', data.time.min()*1e6, 'to', data.time.max()*1e6, 'us')
print()
print('Flame velocity range:', data.flame_velocity.min(), 'to', data.flame_velocity.max(), 'm/s')
print('Burned gas velocity range:', data.burned_gas_velocity.min(), 'to', data.burned_gas_velocity.max(), 'm/s')
print()
print('First 20 points:')
print('  t (us)   v_flame   v_burned   diff')
for i in range(min(20, len(data.time))):
    t = data.time[i]*1e6
    vf = data.flame_velocity[i]
    vb = data.burned_gas_velocity[i]
    print(f'  {t:7.1f}   {vf:7.1f}   {vb:7.1f}   {vb-vf:7.1f}')

print()
print('Average difference (burned - flame):', (data.burned_gas_velocity - data.flame_velocity).mean())
print('Min difference:', (data.burned_gas_velocity - data.flame_velocity).min())
print('Max difference:', (data.burned_gas_velocity - data.flame_velocity).max())
