function [summaryTable, simResults] = run_matlab_comparison(nDays, seed, useSimulink)
% RUN_MATLAB_COMPARISON Evaluates and compares Baseline Thermostat vs. Carbon-Aware MPC in MATLAB.
% Fulfills MathWorks Challenge Requirement 6: Computes energy (kWh), carbon (kg CO2),
% and ASHRAE compliance, producing the comparison table and publication-grade figure.
%
% Syntax:
%   run_matlab_comparison()
%   run_matlab_comparison(nDays)
%   run_matlab_comparison(nDays, seed)
%   run_matlab_comparison(nDays, seed, useSimulink)
%   [summaryTable, simResults] = run_matlab_comparison(...)
%
% Inputs:
%   nDays       - Simulation horizon in days (default: 5)
%   seed        - Random seed for scenario generation (default: 999)
%   useSimulink - Boolean flag to run through Simulink if available (default: false)
%
% Outputs:
%   summaryTable - MATLAB table containing comparative performance metrics
%   simResults   - Struct containing full timeseries trajectories for both controllers
%
% Files generated:
%   - results/matlab_comparison_results.csv
%   - results/matlab_baseline_vs_mpc.png
%
% See also: mpc_controller, create_datacenter_nlmpc, generate_environment, generate_workload

if nargin < 1 || isempty(nDays), nDays = 5; end
if nargin < 2 || isempty(seed), seed = 999; end
if nargin < 3 || isempty(useSimulink), useSimulink = false; end

fprintf('=================================================================\n');
fprintf('  MATLAB CLOSED-LOOP COMPARISON: BASELINE vs. CARBON-AWARE MPC   \n');
fprintf('  Evaluation Horizon: %d days | Scenario Seed: %d             \n', nDays, seed);
fprintf('=================================================================\n\n');

% Set random seed
rng(seed);

% Simulation parameters
dtMinutes = 5.0;
dtHr = dtMinutes / 60.0;
nSteps = round(nDays * 24 * 60 / dtMinutes);
timeHr = (0:nSteps-1)' * dtHr;

% Physical parameters (MUST match ThermalPlant class attributes)
C_ROOM = 50.0;       % Room thermal capacitance (kWh/°C)
UA_ENV = 4.0;        % Envelope heat transfer (kW/°C)
Q_IDLE = 150.0;      % Server idle heat (kW)
Q_IT_MAX = 500.0;    % Server max IT heat (kW)
FAN_PEN_MAX = 30.0;  % Server fan ramp max penalty (kW)
FAN_RAMP_START = 25.0;% Server fan ramp onset (°C)
FAN_RAMP_FULL = 32.0; % Server fan ramp saturation (°C)
Q_COOL_MAX = 600.0;  % Chiller max capacity (kW)
P_FAN_MAX = 40.0;    % Max CRAC fan electrical power (kW)
COP_CAP = 8.5;       % Chiller COP max ceiling
COP_FLOOR = 2.5;     % Chiller COP minimum floor
COP_SLOPE = 0.15;    % COP ambient temperature degradation slope
COP_REF_T = 10.0;    % COP reference ambient temperature (°C)
T_SET = 22.0;        % Initial room temp (°C)
T_REC_LO = 18.0;     % ASHRAE Recommended lower (°C)
T_REC_HI = 27.0;     % ASHRAE Recommended upper (°C)
T_ALW_LO = 15.0;     % ASHRAE Allowable lower (°C)
T_ALW_HI = 32.0;     % ASHRAE Allowable upper (°C)

% Generate scenario profiles
fprintf('Generating deterministic scenario profiles...\n');
envTS = generate_environment(nDays, dtMinutes, []);
workloadTS = generate_workload(nDays, dtMinutes, 0.15, []);

tamb = envTS.Data(1:nSteps);
util = workloadTS.Data(1:nSteps);

% Diurnal carbon intensity model: peaks in evening (17-21h), low at midday solar
% Base: 380 gCO2/kWh, diurnal amplitude: 120 gCO2/kWh, solar dip: -80 gCO2/kWh
hourOfDay = mod(timeHr, 24);
carbonIntensity = 380.0 + 120.0 * sin(2 * pi * (hourOfDay - 14) / 24) ...
    - 80.0 * exp(-((hourOfDay - 12) / 3.5).^2);
carbonIntensity = max(180.0, carbonIntensity);

% Load or instantiate surrogate for MPC
surrogate = [];
try
    surrogate = import_classical_surrogate();
catch
    % Graceful fallback: MPC uses internal forward physics model
end

% -------------------------------------------------------------------------
% 1. RUN BASELINE THERMOSTAT (Relay with hysteresis)
% -------------------------------------------------------------------------
fprintf('Simulating Controller 1: Baseline Thermostat (Relay hysteresis)...\n');
T_base = zeros(nSteps, 1);
u_base = zeros(nSteps, 1);
T_base(1) = T_SET;
thermo_state = false; % Cooling state: false = low (0.25), true = high (1.0)

for k = 1:nSteps
    Tk = T_base(k);
    % Stateflow/Relay logic: Switch ON >= 23°C, Switch OFF <= 21°C
    if Tk >= 23.0
        thermo_state = true;
    elseif Tk <= 21.0
        thermo_state = false;
    end
    
    if thermo_state
        u_base(k) = 1.0;
    else
        u_base(k) = 0.25;
    end
    
    if k < nSteps
        T_base(k+1) = rk4_step(Tk, u_base(k), util(k), tamb(k), dtHr, ...
            C_ROOM, Q_IDLE, Q_IT_MAX, FAN_PEN_MAX, FAN_RAMP_START, FAN_RAMP_FULL, Q_COOL_MAX, UA_ENV);
    end
end

% -------------------------------------------------------------------------
% 2. RUN CARBON-AWARE MPC CONTROLLER
% -------------------------------------------------------------------------
fprintf('Simulating Controller 2: Carbon-Aware Receding-Horizon MPC...\n');
T_mpc = zeros(nSteps, 1);
u_mpc = zeros(nSteps, 1);
T_mpc(1) = T_SET;
u_prev = 0.45;
HORIZON = 24;

opts = struct();
opts.horizon = HORIZON;
opts.dt_minutes = dtMinutes;

for k = 1:nSteps
    Tk = T_mpc(k);
    h = min(HORIZON, nSteps - k + 1);
    
    util_fore = util(k:k+h-1);
    tamb_fore = tamb(k:k+h-1);
    carbon_fore = carbonIntensity(k:k+h-1);
    
    % Pad if near end of scenario
    if h < HORIZON
        util_fore = [util_fore; repmat(util_fore(end), HORIZON - h, 1)];
        tamb_fore = [tamb_fore; repmat(tamb_fore(end), HORIZON - h, 1)];
        carbon_fore = [carbon_fore; repmat(carbon_fore(end), HORIZON - h, 1)];
    end
    
    surrogate_fn = [];
    if ~isempty(surrogate)
        surrogate_fn = surrogate.predict;
    end
    
    % Solve MPC
    u_opt = mpc_controller(Tk, util_fore, tamb_fore, carbon_fore, u_prev, surrogate_fn, opts);
    u_mpc(k) = u_opt;
    u_prev = u_opt;
    
    if ~isempty(surrogate)
        cop_k = min(max(COP_CAP - COP_SLOPE * (tamb(k) - COP_REF_T), COP_FLOOR), COP_CAP);
        pcool_k = P_FAN_MAX * (u_opt^3) + (u_opt * Q_COOL_MAX) / cop_k;
        surrogate.update(Tk, util(k), tamb(k), pcool_k, carbonIntensity(k));
    end
    
    if k < nSteps
        T_mpc(k+1) = rk4_step(Tk, u_mpc(k), util(k), tamb(k), dtHr, ...
            C_ROOM, Q_IDLE, Q_IT_MAX, FAN_PEN_MAX, FAN_RAMP_START, FAN_RAMP_FULL, Q_COOL_MAX, UA_ENV);
    end
    
    if mod(k, 288) == 0
        fprintf('  Day %d / %d completed...\n', round(k / 288), nDays);
    end
end

% -------------------------------------------------------------------------
% 3. COMPUTE METRICS & SAVINGS
% -------------------------------------------------------------------------
% Power and Energy calculation
cop_base = min(max(COP_CAP - COP_SLOPE * (tamb - COP_REF_T), COP_FLOOR), COP_CAP);
p_cool_base = P_FAN_MAX * (u_base.^3) + (u_base * Q_COOL_MAX) ./ cop_base;
energy_base_cum = cumsum(p_cool_base * dtHr);
carbon_base_cum = cumsum(p_cool_base .* carbonIntensity * dtHr) / 1000.0;

cop_mpc = min(max(COP_CAP - COP_SLOPE * (tamb - COP_REF_T), COP_FLOOR), COP_CAP);
p_cool_mpc = P_FAN_MAX * (u_mpc.^3) + (u_mpc * Q_COOL_MAX) ./ cop_mpc;
energy_mpc_cum = cumsum(p_cool_mpc * dtHr);
carbon_mpc_cum = cumsum(p_cool_mpc .* carbonIntensity * dtHr) / 1000.0;

total_energy_base = energy_base_cum(end);
total_energy_mpc = energy_mpc_cum(end);
total_carbon_base = carbon_base_cum(end);
total_carbon_mpc = carbon_mpc_cum(end);

energy_savings_pct = 100.0 * (total_energy_base - total_energy_mpc) / total_energy_base;
carbon_savings_pct = 100.0 * (total_carbon_base - total_carbon_mpc) / total_carbon_base;

rec_compliance_base = 100.0 * mean(T_base >= T_REC_LO & T_base <= T_REC_HI);
rec_compliance_mpc = 100.0 * mean(T_mpc >= T_REC_LO & T_mpc <= T_REC_HI);

alw_compliance_base = 100.0 * mean(T_base >= T_ALW_LO & T_base <= T_ALW_HI);
alw_compliance_mpc = 100.0 * mean(T_mpc >= T_ALW_LO & T_mpc <= T_ALW_HI);

% Compressor cycles (relay switch transitions)
cycles_base = sum(abs(diff(u_base > 0.5)));
cycles_mpc = sum(abs(diff(u_mpc > 0.5)));

% Construct summary table
Controller = {'Baseline Thermostat'; 'Carbon-Aware MPC'; 'Relative Improvement'};
Energy_kWh = [total_energy_base; total_energy_mpc; -energy_savings_pct];
Carbon_kg = [total_carbon_base; total_carbon_mpc; -carbon_savings_pct];
ASHRAE_Rec_Pct = [rec_compliance_base; rec_compliance_mpc; rec_compliance_mpc - rec_compliance_base];
ASHRAE_Alw_Pct = [alw_compliance_base; alw_compliance_mpc; alw_compliance_mpc - alw_compliance_base];
Max_Temp_C = [max(T_base); max(T_mpc); max(T_mpc) - max(T_base)];
Min_Temp_C = [min(T_base); min(T_mpc); min(T_mpc) - min(T_base)];
Compressor_Cycles = [cycles_base; cycles_mpc; cycles_mpc - cycles_base];

summaryTable = table(Controller, Energy_kWh, Carbon_kg, ASHRAE_Rec_Pct, ...
    ASHRAE_Alw_Pct, Max_Temp_C, Min_Temp_C, Compressor_Cycles);

fprintf('\n=================================================================\n');
fprintf('                     SUMMARY OF RESULTS                          \n');
fprintf('=================================================================\n');
disp(summaryTable);
fprintf('Key Takeaway: MPC achieves %.1f%% Cooling Energy & %.1f%% CO2 Reduction\n', ...
    energy_savings_pct, carbon_savings_pct);
fprintf('ASHRAE Recommended Compliance: %.2f%% (MPC) vs %.2f%% (Baseline)\n\n', ...
    rec_compliance_mpc, rec_compliance_base);

% Save CSV
repoRoot = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoRoot, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
csvFile = fullfile(resultsDir, 'matlab_comparison_results.csv');
writetable(summaryTable, csvFile);
fprintf('Saved summary metrics to: %s\n', csvFile);

% -------------------------------------------------------------------------
% 4. GENERATE PUBLICATION-QUALITY FIGURE
% -------------------------------------------------------------------------
fig = figure('Name', 'Data Center Cooling: Baseline vs Carbon-Aware MPC', ...
    'Units', 'pixels', 'Position', [100, 100, 1100, 850], 'Visible', 'off');

% Panel 1: Temperature trajectories
subplot(3, 1, 1);
hold on; grid on; box on;
% Shaded ASHRAE bands
patch([timeHr(1), timeHr(end), timeHr(end), timeHr(1)], ...
    [T_ALW_LO, T_ALW_LO, T_ALW_HI, T_ALW_HI], [0.98, 0.95, 0.85], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'ASHRAE Allowable (15-32°C)');
patch([timeHr(1), timeHr(end), timeHr(end), timeHr(1)], ...
    [T_REC_LO, T_REC_LO, T_REC_HI, T_REC_HI], [0.85, 0.95, 0.85], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'ASHRAE Recommended (18-27°C)');

plot(timeHr, T_base, 'Color', [0.85, 0.30, 0.10], 'LineWidth', 1.4, 'DisplayName', 'Baseline Thermostat');
plot(timeHr, T_mpc, 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.6, 'DisplayName', 'Carbon-Aware MPC');
yline(22.0, 'k--', 'LineWidth', 1.0, 'DisplayName', 'Target Setpoint (22°C)');
ylabel('Room Temp (°C)', 'FontWeight', 'bold');
title('(a) Thermal Trajectory & ASHRAE Standard Compliance', 'FontWeight', 'bold');
xlim([0, timeHr(end)]);
ylim([14, 30]);
legend('Location', 'northeast', 'NumColumns', 3);

% Panel 2: Control actuation and carbon intensity
subplot(3, 1, 2);
yyaxis left;
hold on; grid on; box on;
plot(timeHr, u_base, 'Color', [0.85, 0.30, 0.10, 0.6], 'LineWidth', 1.2, 'DisplayName', 'Baseline u_{cool}');
plot(timeHr, u_mpc, 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.5, 'DisplayName', 'MPC u_{cool}');
ylabel('Cooling Command u \in [0, 1]', 'FontWeight', 'bold');
ylim([-0.05, 1.15]);

yyaxis right;
plot(timeHr, carbonIntensity, 'Color', [0.45, 0.20, 0.60], 'LineStyle', ':', 'LineWidth', 1.5, ...
    'DisplayName', 'Carbon Intensity (gCO_2/kWh)');
ylabel('Carbon Intensity (gCO_2/kWh)', 'FontWeight', 'bold');
title('(b) Actuation Response & Grid Carbon Tracking (Pre-Cooling Dynamics)', 'FontWeight', 'bold');
xlim([0, timeHr(end)]);

% Panel 3: Cumulative carbon and energy savings
subplot(3, 1, 3);
yyaxis left;
hold on; grid on; box on;
plot(timeHr, carbon_base_cum, 'Color', [0.85, 0.30, 0.10], 'LineWidth', 1.4, 'DisplayName', 'Baseline CO_2');
plot(timeHr, carbon_mpc_cum, 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.6, 'DisplayName', 'MPC CO_2');
ylabel('Cum. Carbon (kg CO_2)', 'FontWeight', 'bold');

yyaxis right;
plot(timeHr, energy_base_cum, 'Color', [0.85, 0.30, 0.10], 'LineStyle', '--', 'LineWidth', 1.2, 'DisplayName', 'Baseline Energy');
plot(timeHr, energy_mpc_cum, 'Color', [0.00, 0.45, 0.74], 'LineStyle', '--', 'LineWidth', 1.5, 'DisplayName', 'MPC Energy');
ylabel('Cum. Energy (kWh)', 'FontWeight', 'bold');
xlabel('Simulation Time (hours)', 'FontWeight', 'bold');
title(sprintf('(c) Environmental Impact: -%.1f%% Carbon Emissions & -%.1f%% Cooling Energy', ...
    carbon_savings_pct, energy_savings_pct), 'FontWeight', 'bold');
xlim([0, timeHr(end)]);

pngFile = fullfile(resultsDir, 'matlab_baseline_vs_mpc.png');
saveas(fig, pngFile);
close(fig);
fprintf('Saved comparison figure to: %s\n', pngFile);

% Package outputs
simResults = struct();
simResults.timeHr = timeHr;
simResults.T_base = T_base;
simResults.u_base = u_base;
simResults.T_mpc = T_mpc;
simResults.u_mpc = u_mpc;
simResults.carbonIntensity = carbonIntensity;
simResults.p_cool_base = p_cool_base;
simResults.p_cool_mpc = p_cool_mpc;

end

% Local RK4 integration step function
function T_next = rk4_step(T, u, util, tamb, dt_hr, C_room, Q_idle, Q_it_max, fan_pen_max, fan_ramp_start, fan_ramp_full, Q_cool_max, UA_env)
    function dT = thermal_deriv(Temp)
        ramp = min(max((Temp - fan_ramp_start) / (fan_ramp_full - fan_ramp_start), 0.0), 1.0);
        qit = Q_idle + (Q_it_max - Q_idle) * util + fan_pen_max * (ramp^2);
        q_cool = u * Q_cool_max;
        q_env = UA_env * (tamb - Temp);
        dT = (qit - q_cool + q_env) / C_room; % degC / hr
    end

    k1 = thermal_deriv(T);
    k2 = thermal_deriv(T + 0.5 * dt_hr * k1);
    k3 = thermal_deriv(T + 0.5 * dt_hr * k2);
    k4 = thermal_deriv(T + dt_hr * k3);
    
    T_next = T + (dt_hr / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
end
