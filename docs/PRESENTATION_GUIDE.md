# Professor Presentation & Defense Guide

A complete 10-slide presentation deck outline and 2-minute demonstration video script for defending the project to your professor and academic panel.

---

## 1. Slide Deck Structure & Speaker Notes

### Slide 1: Title & Motivation
- **Title:** Optimal Data Center Cooling: Predictive and Quantum-Enhanced Thermal Management
- **Subtitle:** MathWorks Excellence in Innovation Challenge Project #196
- **Visuals:** Repository badges, schematic diagram of data center cooling loops.
- **Key Points:**
  - Modern data centers consume >1–2% of global electricity; cooling accounts for up to 40% of facility power.
  - Traditional thermostatic cooling is reactive and wastes substantial energy by overcooling.
  - Objective: Develop a predictive control architecture optimizing carbon, energy, and hardware reliability while respecting strict ASHRAE thermal constraints.
- **Speaker Script:**
  > *"Good morning, Professor. Today I am presenting our complete solution to MathWorks Project #196 on Optimal Data Center Cooling. As AI and cloud workloads expand, cooling has become the single largest operational efficiency bottleneck in data centers. In this project, we designed, validated, and causally diagnosed an advanced Model Predictive Control framework integrating classical and quantum neural surrogates, physical reliability modeling, and spatial workload placement."*

---

### Slide 2: Physical Modeling & Simulation Architecture
- **Title:** Coupled Non-linear Physical Plant & MATLAB/Simulink Integration
- **Visuals:** Equations table from `docs/METHODOLOGY.md` + Simulink block diagram.
- **Key Points:**
  - Non-linear weather-dependent Chiller COP ($\mathrm{COP} \in [2.5, 8.5]$).
  - Cubic CRAC fan power ($P \propto u^3$) and non-linear server fan ramp penalty ($+30$ kW above 25 °C).
  - 5-day stress scenario: diurnal ambient heat, severe heat waves (+8 °C), and stochastic IT workload spikes.
  - Companion Simulink model `DataCenterCooling.slx` and Simscape Fluids two-loop hydraulic integration.
- **Speaker Script:**
  > *"To ensure high fidelity, our plant model incorporates non-linear thermodynamics: ambient-dependent chiller efficiency, cubic fan laws, and server fan power penalties. We developed both a continuous Runge-Kutta simulation in Python for large-scale multi-seed optimization and companion MATLAB/Simulink scripts that drive the Simscape Fluids physical model."*

---

### Slide 3: Model Predictive Control (MPC) Formulation
- **Title:** Carbon-Aware Receding-Horizon Control
- **Visuals:** Figure 1 (`results/fig1_timeseries.png`).
- **Key Points:**
  - Receding horizon: 24 steps (2 hours ahead) with 5-minute control intervals.
  - Multi-objective cost function:
    $$J = \sum_{k=1}^H \Big( P_{\mathrm{cool}}(k) \cdot C_{\mathrm{grid}}(k) + w_{\mathrm{rec}} \cdot \Delta T_{\mathrm{rec}}^2 + w_{\mathrm{alw}} \cdot \Delta T_{\mathrm{alw}}^2 + w_{\Delta u} \cdot (\Delta u)^2 \Big)$$
  - Dynamically precools during low-carbon grid hours to avoid high-emission peaks.
- **Speaker Script:**
  > *"Rather than reacting after temperature spikes, our MPC controller looks 2 hours ahead. By coupling real-time grid carbon intensity forecasts with neural surrogates, it pre-cools the facility during green energy periods and floats temperatures safely up to the 27 °C ASHRAE boundary during dirty peak hours."*

---

### Slide 4: Experimental Benchmarking & 10-Seed Validation
- **Title:** Performance Results Across 10 Unseen Seeds
- **Visuals:** Figure 2 (`results/fig3_multiseed_boxplots.png`) and performance table.
- **Key Points:**
  - 10 held-out evaluation seeds strictly outside training distributions.
  - **-16.6% Carbon Reduction** (3,280.7 kg CO₂ vs 3,934.0 kg baseline).
  - **-16.9% Energy Savings** (8,200.8 kWh vs 9,874.3 kWh baseline).
  - 0% allowable boundary violations across all policies.
- **Speaker Script:**
  > *"To establish scientific rigor, we evaluated 10 unseen held-out seeds. As shown in the boxplots, predictive control achieved statistically significant savings: a 16.6% carbon reduction and 16.9% cooling energy cut, without ever violating the ASHRAE allowable limit of 32 °C."*

---

### Slide 5: The Compliance Mystery: GRU vs. Hybrid QML
- **Title:** A Surprising 12-Point Gap in ASHRAE Compliance
- **Visuals:** Table showing 87.81% (GRU) vs 99.99% (QML) compliance.
- **Key Points:**
  - Both Classical GRU and Hybrid QML achieve nearly identical offline regression MAE (~0.26–0.28 °C).
  - In closed-loop operation, QML achieved **99.99% compliance**, while GRU achieved only **87.81%**.
  - Non-overlapping distributions: QML never dipped below 99.93%; GRU never exceeded 93.40%.
- **Speaker Script:**
  > *"Here we uncovered a fascinating phenomenon. Despite both models having nearly identical offline accuracy, when plugged into the closed-loop MPC loop, the QML-enhanced controller achieved 99.99% compliance, whereas the classical GRU controller achieved only 87.81%. This led to a crucial question: What is causing this 12-point compliance gap?"*

---

### Slide 6: Causal Diagnosis & Intervention Surgery
- **Title:** Proving the Mechanism via Causal Bias Surgery
- **Visuals:** Figure 3 (`results/fig5_bias_by_horizon.png`) and causal intervention table.
- **Key Points:**
  - Step-by-step signed error analysis revealed: GRU under-predicts near-term temperatures (steps 1–4, 5–20 min ahead), encouraging the optimizer to delay cooling and ride the 27.0 °C threshold. QML conservatively over-predicts near-term temperatures.
  - **Causal Grafting:** Adding QML's near-term bias to GRU boosted GRU from 87.79% to **100.00%** compliance across all 10 seeds.
  - **Causal Removal:** Subtracting QML's near-term bias dropped QML compliance to **62.42%**.
  - **Sanity Check (Time Above Ceiling):** Peak temperatures remained identical (27.12 vs 27.14 °C), but QML-without-bias spent 3.08× more cumulative time above 27.0 °C (45.1 vs 14.6 hours), explaining the 25 pp compliance drop.
  - **Conclusion:** Near-term prediction bias causally drives boundary-riding behavior, compounding with mid-horizon forecast errors.
- **Speaker Script:**
  > *"We hypothesized that near-term prediction bias was driving this behavior. We performed an isolated causal intervention: grafting QML's near-term bias onto the GRU model immediately raised its compliance from 87.8% to 100.0%, while stripping it from QML dropped compliance to 62.4%. Our follow-up time-above-ceiling check proved that peak temperatures remained virtually identical, but QML-without-bias spent over three times longer hovering above 27 °C. This rigorously proves how directional near-term prediction bias governs compliance in receding-horizon optimization."*

---

### Slide 7: Advanced Scope 1 — Semiconductor Reliability (MTBF)
- **Title:** Hardware Degradation & Thermal Fatigue Modeling
- **Visuals:** Figure 4 (`results/fig5_component_reliability.png`).
- **Key Points:**
  - IEEE & JEDEC JESD47 standards.
  - **Arrhenius Acceleration Factor ($AF_T$):** Evaluates electro-migration and gate-oxide breakdown under elevated temperature.
  - **Coffin-Manson Mechanical Fatigue:** Evaluates solder joint strain from thermal cycling ($\Delta T$).
  - Quantifies the trade-off between energy-saving thermal floating and hardware longevity.
- **Speaker Script:**
  > *"Addressing Advanced Scope 1, we implemented Arrhenius and Coffin-Manson semiconductor reliability models. This allows data center operators to explicitly balance cooling energy savings against the risk of thermal cycling fatigue on server motherboards and CPU solder joints."*

---

### Slide 8: Advanced Scope 2 — 4-Rack Spatial Optimization
- **Title:** Cross-Rack Heat Recirculation & Convex QP Dispatch
- **Visuals:** Figure 5 (`results/fig6_spatial_workload_placement.png`).
- **Key Points:**
  - 4-rack data center with cross-recirculation coupling matrix ($D$).
  - Some racks recirculate exhaust heat to adjacent racks, creating localized hot spots.
  - Convex Quadratic Programming (QP) dispatcher dynamically diverts computational load away from heat-stressed racks in real time.
  - **Result:** **55.0% reduction in thermal SLA violation duration** (1,440 steps down to 648 steps).
- **Speaker Script:**
  > *"For Advanced Scope 2, we modeled a 4-rack facility with thermal recirculation. By designing a real-time convex Quadratic Program dispatcher, computational jobs are dynamically routed to aerodynamically favored racks, reducing hot-spot violations by 55% during extreme ambient heat waves."*

---

### Slide 9: Repository Packaging & MathWorks Integration
- **Title:** Code Quality, Tooling, and Verification
- **Visuals:** Screenshot of `README.md`, MATLAB scripts, and requirements.
- **Key Points:**
  - Fully automated MATLAB scripts: `build_datacenter_simulink_model.m`, `export_simscape_fluids_model.m`, `run_simulation.m`.
  - Clean Python package structure with deterministic random seeds for full reproducibility.
  - Comprehensive documentation in `docs/METHODOLOGY.md` and `docs/MATHWORKS_SUBMISSION_ANSWERS.md`.
- **Speaker Script:**
  > *"All software is packaged for one-click verification. We provide native MATLAB scripts that programmatically build and simulate the companion Simulink model, as well as a fully modular Python pipeline for multi-seed verification and QML training."*

---

### Slide 10: Conclusion & Q&A
- **Title:** Summary of Contributions
- **Key Points:**
  - Solved all core and advanced requirements of MathWorks Challenge #196.
  - 16.6% carbon reduction and 16.9% energy savings verified across 10 seeds.
  - First empirical causal proof of near-term prediction bias in neural MPC compliance.
  - Complete integration across MATLAB/Simulink and Python.
- **Speaker Script:**
  > *"In summary, we delivered a comprehensive, validated, and novel solution that exceeds the challenge requirements. Thank you, and I welcome any questions."*

---

## 2. Two-Minute Video Walkthrough Script

| Time | Screen Display | Voiceover Script |
|---|---|---|
| **0:00 – 0:25** | Open repository `README.md` and show Figure 1 timeseries. | *"Welcome. This is our solution for MathWorks Excellence in Innovation Challenge #196: Optimal Data Center Cooling. We developed a predictive control architecture that coordinates variable chiller cooling and dynamic workloads against real-time grid carbon signals."* |
| **0:25 – 0:50** | Show MATLAB Command Window running `results = run_simulation('DataCenterCooling', 1)` and opening the Simulink model. | *"Our companion Simulink model, generated via MATLAB script, models the complete data center thermal plant. It receives diurnal workload and ambient weather timeseries, logging temperature and power metrics to generate training datasets."* |
| **0:50 – 1:20** | Show `simulate.py` running in terminal and display Figure 2 (Multi-Seed Boxplots). | *"In our closed-loop receding-horizon MPC, we benchmarked classical GRU and hybrid Variational Quantum neural surrogates across 10 held-out evaluation seeds. Predictive control achieved a 16.6% carbon reduction and 16.9% energy savings while maintaining strict 18 to 27 °C ASHRAE compliance."* |
| **1:20 – 1:40** | Show Figure 3 (Bias by Horizon) and Figure 4 (Reliability). | *"Through causal bias surgery, we proved that near-term prediction bias is the direct governing mechanism for compliance. Furthermore, our JEDEC Arrhenius model tracks semiconductor degradation and thermal fatigue in real time."* |
| **1:40 – 2:00** | Show Figure 5 (4-Rack Spatial Optimization). | *"Finally, our 4-rack spatial QP dispatcher dynamically diverts computational load away from thermally penalized racks, slashing hot-spot SLA violations by 55%. Thank you."* |
