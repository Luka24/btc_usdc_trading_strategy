# INDEX - Project Navigation Guide

## Quick Navigation

### For Beginners
Start here → [00_START_HERE.md](00_START_HERE.md) (5 min)

### For Technical Overview
→ [README.md](README.md) (20 min)

### For Data Sources & Methodology ⭐ NEW
→ [SOURCES.md](SOURCES.md) - Detailed data sources and how parameters were calculated (15 min)

### For Results
→ [results/](results/) folder - All outputs (CSV, TXT, PNG)

### For Python Code (Core Strategy)
→ [config.py](config.py) - Configuration parameters  
→ [production_cost.py](production_cost.py) - BTC cost calculation  
→ [signal_generator.py](signal_generator.py) - Trading signals  
→ [portfolio.py](portfolio.py) - Portfolio management  
→ [backtest.py](backtest.py) - Backtesting engine  
→ [risk_manager.py](risk_manager.py) - Risk controls  
→ [main.py](main.py) - Main orchestration

### For AI/ML Parameter Optimization ⭐ NEW
**Start with this for parameter tuning!**

→ [optimization/](optimization/) - **All optimization tools and documentation here**
  - [README.md](optimization/README.md) - Module overview
  - [00_READ_ME_OPTIMIZATION.txt](optimization/00_READ_ME_OPTIMIZATION.txt) - Start here! (5 min)
  - [grid_search_optimizer.py](optimization/grid_search_optimizer.py) - Fast search (5 min)
  - [parameter_optimizer.py](optimization/parameter_optimizer.py) - Bayesian (20 min) ⭐
  - [genetic_optimizer.py](optimization/genetic_optimizer.py) - Genetic algorithm (45 min)
  - [run_all_optimizers.py](optimization/run_all_optimizers.py) - Compare all methods
  - [visualization_tool.py](optimization/visualization_tool.py) - Analysis plots

### For Interactive Demo
→ [strategy_notebook.ipynb](strategy_notebook.ipynb) - Interactive Jupyter notebook

---

## File Overview

| File | Language | Purpose | Time |
|------|----------|---------|------|
| 00_START_HERE.md | English | Quick start guide | 5 min |
| README.md | English | Technical overview | 20 min |
| INDEX.md | English | This navigation file | 2 min |
| **optimization/** | **Python + MD** | **AI/ML parameter optimization module** | **5-45 min** |
| config.py | English | Configuration parameters | Read |
| production_cost.py | English | BTC cost calculation | Read |
| signal_generator.py | English | Trading signals | Read |
| portfolio.py | English | Portfolio management | Read |
| backtest.py | English | Backtesting engine | Read |
| risk_manager.py | English | Risk controls | Read |
| main.py | English | Main orchestration | Run |
| results/ | - | Output folder | Review |

---

## Results Folder Structure

```
results/
├── backtest_results_YYYYMMDD_HHMMSS.csv    # Backtest data
├── strategy_report_YYYYMMDD_HHMMSS.txt     # Analysis reports
└── strategy_comprehensive_analysis.png      # Visualizations
```

All outputs from `python main.py` are automatically saved to the `results/` folder.

---

## Documentation Files

### 00_START_HERE.md
- Quick start instructions
- Project overview
- Interview talking points
- File structure
- How to use the system

### README.md
- Complete technical guide
- Component descriptions
- Mathematical foundations
- Configuration options
- Design rationale
- Potential improvements

### INDEX.md (This File)
- Project navigation
- File descriptions
- Quick access links

---

## How to Use

### Quick Start
1. Start with: [00_START_HERE.md](00_START_HERE.md)
2. Read: [README.md](README.md)
3. Run: `python main.py`
4. Review: `results/` folder
5. Explore: [strategy_notebook.ipynb](strategy_notebook.ipynb)

### For Development
1. Open: [config.py](config.py) - Adjust parameters
2. Understand: System architecture from [README.md](README.md)
3. Run: `python main.py` - Generate results
4. Deep dive: Other Python modules
5. Iterate: Modify and re-run

---

## Project Status

✅ **Complete - Production Ready**
- All Python code: 100% English
- All documentation: 100% English
- Results organized in dedicated folder
- System fully tested and operational

---

## Quick Reference Commands

### Run Complete System
```bash
python main.py
```

### Interactive Demo
```bash
jupyter notebook strategy_notebook.ipynb
```

### Check Configuration
```python
python -c "from config import ProductionCostConfig; print(ProductionCostConfig.ENERGY_PRICE_USD_PER_KWH)"
```

### Test All Imports
```bash
python -c "import config, production_cost, signal_generator, portfolio, backtest, risk_manager; print('All modules OK')"
```

### View Results
```bash
# Windows
explorer results

# Or list files
dir results
```

---

**Status**: Production Ready  
**Language**: 100% English  
**Total Files**: 3 markdown + 7 Python + 1 Jupyter  
**Total Code**: 2,500+ lines  
**Results**: Organized in `results/` folder

👉 **START HERE**: [00_START_HERE.md](00_START_HERE.md)

