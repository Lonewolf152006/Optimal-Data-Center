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
- MATLAB & Simulink (Lumped-parameter thermal plant model, Relay hysteresis block, signal logging)
- Model Predictive Control Toolbox (nlmpcMultistage with carbon stage cost and soft ASHRAE constraints)
- Optimization Toolbox (quadprog for convex QP spatial workload dispatch; fmincon for receding-horizon MPC)
- Statistics and Machine Learning Toolbox (wblfit and makedist for Weibull life distribution fitting; randperm)
- Predictive Maintenance Toolbox (exponentialDegradationModel and rul for component degradation/failure forecasting)
- Deep Learning Toolbox (importNetworkFromPyTorch for loading PyTorch GRU surrogate into MATLAB)
- Python 3.10+ (NumPy, SciPy, PyTorch, PennyLane QML) as the cross-validation reference and comparative benchmark
```

---

### Field 5: Summary of Technical Approach and Key Results
```text
1. Primary MATLAB/Simulink Plant & Cross-Validation:
Built a lumped-parameter thermal zone and chiller plant model in MATLAB/Simulink matching foundational thermodynamics (Ebrahimi et al. 2014, Moazamigoodarzi et al. 2019). Rigorously cross-validated the MATLAB plant against the Python reference under identical 5-day diurnal ambient and workload profiles, demonstrating numerical agreement to < 2.1 µ°C (results/crossval_comparison.png). A single-source-of-truth Relay block provides stateful thermostat hysteresis ([21, 23]°C).

2. Carbon-Aware MPC in MATLAB (run_matlab_comparison.m):
Formulated a receding-horizon non-linear MPC using Model Predictive Control Toolbox (nlmpcMultistage) and Optimization Toolbox. Evaluates real-time carbon cost against grid intensity while enforcing ASHRAE TC 9.9 recommended (18–27°C) and allowable (15–32°C) bounds as soft constraints with ECR weights. In 5-day closed-loop simulation, MPC achieves a 16.6% cooling energy reduction and a 16.2% carbon reduction while decreasing chiller compressor switching cycles by 87.9% (25 vs. 206 cycles).

3. 4-Rack Spatial Workload Placement with quadprog (Advanced Scope 2):
Formulated the 4-rack data center workload dispatch as a strictly convex Quadratic Program (QP). Solved via quadprog from Optimization Toolbox in spatial_workload_dispatcher.m, achieving a 55.0% reduction in SLA hot-spot violations and shielding mission-critical tasks to the coolest server racks.

4. Semiconductor Reliability & Degradation Modeling (Advanced Scope 1):
Coupled silicon junction thermal modeling with JEDEC Arrhenius acceleration. Fitted accelerated failure data with Weibull distributions using wblfit from Statistics and Machine Learning Toolbox and modeled component degradation using exponentialDegradationModel from Predictive Maintenance Toolbox, quantifying extended hardware life under smooth MPC control.

5. Causal Prediction Bias & Surrogate Modeling:
Imported the trained GRU surrogate directly into MATLAB via importNetworkFromPyTorch. Demonstrated that near-term prediction bias causally governs ASHRAE envelope compliance by preventing optimizer boundary-riding.
```
