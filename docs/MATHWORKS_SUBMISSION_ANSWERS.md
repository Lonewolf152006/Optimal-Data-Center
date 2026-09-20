# MathWorks Excellence in Innovation Submission Form — Ready-to-Paste Draft

**Project Details:**
- **Challenge Title:** Optimal Data Center Cooling
- **Project Number:** 196
- **Submission Portal:** [MathWorks Excellence in Innovation Solution Form](https://www.mathworks.com/academia/student-challenge/mathworks-excellence-in-innovation-submission-form.html?tfa_1=Optimal%20Data%20Center%20Cooling&tfa_2=196)

---

### Field 1: Project Title
```text
Optimal Data Center Cooling: Predictive and Quantum-Enhanced Thermal Management with Spatial Workload Optimization
```

---

### Field 2: Repository URL
```text
https://github.com/Lonewolf152006/Optimal-Data-Center
```

---

### Field 3: Project Abstract (250 words)
```text
Data center cooling infrastructure accounts for up to 40% of total facility energy, presenting a critical sustainability bottleneck for modern high-density compute. This project addresses MathWorks Excellence in Innovation Challenge #196 ("Optimal Data Center Cooling") by delivering an end-to-end physics-informed control framework that balances thermodynamic efficiency, grid carbon intensity, and hardware reliability.

We couple a dynamic, non-linear thermal plant model (incorporating weather-dependent chiller COP, non-linear server fan power curves, and diurnal IT load spikes) with a receding-horizon Model Predictive Controller (MPC). To accelerate prediction while preserving non-linear dynamics, we evaluate both classical Recurrent Neural Networks (GRU) and Variational Quantum Machine Learning (PennyLane QML) neural surrogates across 10 independent held-out evaluation seeds. 

Across rigorous multi-seed testing, predictive control achieved a 16.6% reduction in operational carbon emissions and a 16.9% reduction in total cooling energy compared to standard thermostatic baseline control. Furthermore, causal intervention surgery revealed that near-term signed prediction bias in the neural surrogate is the direct governing mechanism for ASHRAE TC 9.9 recommended envelope compliance (99.99% for QML vs. 87.81% for GRU).

Expanding beyond the core requirements, we fulfilled both advanced scopes: (1) integrating JEDEC/IEEE Arrhenius and Coffin-Manson semiconductor reliability models to evaluate thermal degradation and MTBF, and (2) implementing a 4-rack cross-recirculation spatial thermal model with a convex Quadratic Program dispatcher that reduces critical rack thermal violations by 55.0%. Companion Simulink/Simscape Fluids models and Python optimization code are packaged for reproducible one-command verification.
```

---

### Field 4: MathWorks & External Tools Used
```text
- MATLAB & Simulink (Model-Based Design, signal logging, system integration)
- Simscape / Simscape Fluids (Two-loop data center hydraulic & thermal liquid modeling)
- Optimization Toolbox (Convex quadratic programming, receding horizon optimization)
- Python 3.10+ (NumPy, SciPy, PyTorch)
- PennyLane (Variational quantum neural network simulation)
- Matplotlib & Seaborn (Scientific visualization and statistical verification)
```

---

### Field 5: Summary of Technical Approach and Key Results
```text
1. Dynamic Plant & Controller Integration:
Built upon foundational literature (Ebrahimi et al. 2014, Moazamigoodarzi et al. 2019, Mousavi et al. 2015), creating a dynamic thermal plant and companion Simulink model ('DataCenterCooling.slx') driven by diurnal ambient temperatures and stochastic IT workload spikes. Formulated a receding-horizon MPC framework optimizing cooling actuation against hourly grid carbon intensity while strictly maintaining ASHRAE TC 9.9 thermal bounds.

2. Benchmark & Sustainability Validation (PUE & WUE):
Evaluated performance across 10 unseen held-out seeds (seeds 101–110). MPC reduced 5-day cooling energy from 9,874.3 kWh to 8,200.8 kWh (-16.9%) and carbon emissions from 3,934.0 kg to 3,280.7 kg CO2 (-16.6%), completely eliminating ASHRAE allowable envelope violations. This drives facility Power Usage Effectiveness (PUE) down from 1.238 to 1.198 (-16.9% cooling overhead) and conserves ~3,000+ liters of evaporative cooling tower makeup water per 5-day cycle.

3. Discovery & Causal Proof of Near-Term Bias Mechanism:
Identified that while GRU and QML show comparable offline test MAE (~0.26–0.28 °C), QML achieves 99.99% ASHRAE compliance vs 87.81% for GRU. Signed bias profiling and causal grafting/removal surgery proved that near-term prediction bias (steps 1–4, 5–20 min ahead) causally determines compliance by preventing optimizer boundary-riding.

4. Semiconductor Reliability Modeling (Advanced Scope 1):
Integrated JEDEC JESD85 Arrhenius acceleration and Coffin-Manson low-cycle thermal fatigue models, quantifying the trade-off between energy-saving thermal cycling and semiconductor Mean Time Between Failures (MTBF).

5. Spatial Thermal Coupling & Workload Placement (Advanced Scope 2):
Formulated a 4-rack data center with cross-rack heat recirculation (Tang et al. 2008). Designed a real-time convex QP dispatcher that routes computing jobs to cooler racks during heat waves, achieving a 55.0% reduction in rack SLA thermal violations and flattening localized thermal gradients by 17.0%.
```
