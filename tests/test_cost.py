from production_cost import BTCProductionCostCalculator

calc = BTCProductionCostCalculator()
hashrate = 520  # EH/s (from data)
cost = calc.calculate_total_cost_per_btc(hashrate)

print(f'Hashrate: {hashrate} EH/s')
print(f'Production cost: ${cost:.2f}')
print(f'BTC price (example): $90,000')
print(f'Ratio: {90000/cost:.2f}')

print('\nComponents:')
energy = calc.calculate_energy_cost_per_btc(hashrate)
print(f'  Energy cost: ${energy:.2f}')
print(f'  OPEX (20%): ${energy*0.20:.2f}')
print(f'  Depreciation (15%): ${energy*0.15:.2f}')
print(f'  Total before buffer: ${energy*1.35:.2f}')
print(f'  Total with buffer (5%): ${energy*1.35*1.05:.2f}')
