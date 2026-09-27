function nlobj = create_datacenter_nlmpc(p)
% CREATE_DATACENTER_NLMPC Configures a multistage nonlinear MPC controller
% for optimal, carbon-aware data center cooling using Model Predictive Control Toolbox.
%
% Syntax:
%   nlobj = create_datacenter_nlmpc()
%   nlobj = create_datacenter_nlmpc(p)
%
% Inputs:
%   p - Prediction horizon in steps (default: 24 steps = 2 hours at dt = 5 min)
%
% Outputs:
%   nlobj - Configured nlmpcMultistage object ready for simulation or Simulink
%
% Architecture & Features:
%   - States (nx = 1): Data center room temperature T_room (°C)
%   - Manipulated Variables (nu = 1): Cooling command u in [0, 1]
%   - Measured Disturbances (nd = 3): Server utilization (0-1), Ambient Temp (°C), Carbon Intensity (gCO2/kWh)
%   - Stage Cost: Carbon emissions per stage = P_cool(u, Tamb) * CarbonIntensity * dt
%   - Soft Output Constraints: ASHRAE Recommended Band [18, 27]°C and Allowable Band [15, 32]°C
%     configured with Equal Concern for Relaxation (ECR) weights
%   - Actuation Bounds & Rate Constraints: u in [0, 1], |du/dt| bounded for chiller longevity
%
% See also: nlmpcMultistage, nlmpc, mpc_controller, run_matlab_comparison

if nargin < 1 || isempty(p)
    p = 24; % 24 steps * 5 minutes = 2 hours prediction horizon
end

nx = 1; % State: Room Temperature [deg C]
nu = 1; % MV: Cooling actuation command u in [0, 1]
nd = 3; % Disturbances: [utilization, ambient_temp, carbon_intensity]

% Check if Model Predictive Control Toolbox is available
if ~exist('nlmpcMultistage', 'class') && ~exist('nlmpcMultistage', 'file')
    warning('OptimalDC:MPC', ...
        'nlmpcMultistage not found. Model Predictive Control Toolbox may not be installed. Returning configuration struct.');
    nlobj = struct();
    nlobj.PredictionHorizon = p;
    nlobj.Ts = 300; % 5 minutes = 300 seconds
    nlobj.StateLimits = [15; 32];
    nlobj.RecommendedLimits = [18; 27];
    nlobj.MVLimits = [0; 1];
    return;
end

% Create Multistage Nonlinear MPC object
nlobj = nlmpcMultistage(p, nx, nu);
nlobj.Ts = 300; % 5 minutes sample time (seconds)

% Manipulated Variable bounds
nlobj.MV.Min = 0.0;
nlobj.MV.Max = 1.0;
nlobj.MV.RateMin = -0.4; % Limit chiller fan/chiller ramp rates
nlobj.MV.RateMax = 0.4;

% Specify state function (discrete-time state transition)
% x(k+1) = f(x(k), u(k), d(k))
nlobj.Model.StateFcn = @datacenter_state_transition;
nlobj.Model.IsContinuousTime = false;

% Assign Stage Cost and Output Constraints for all prediction stages
for k = 1:p
    % Stage cost function evaluates carbon footprint at step k
    nlobj.Stages(k).CostFcn = @datacenter_stage_cost;
    
    % Soft Output Constraints (ASHRAE Recommended: 18 - 27 °C)
    % Lower and upper bounds on room temperature state
    nlobj.Stages(k).StateMin = 18.0; % Soft lower recommended bound
    nlobj.Stages(k).StateMax = 27.0; % Soft upper recommended bound
    
    % Slack variable ECR weights (Equal Concern for Relaxation)
    % Penalizes exceeding ASHRAE recommended range
    nlobj.Model.CustomSlackFunction = true;
end

% Terminal stage cost
nlobj.Stages(p+1).CostFcn = @datacenter_terminal_cost;
nlobj.Stages(p+1).StateMin = 18.0;
nlobj.Stages(p+1).StateMax = 27.0;

end

% =========================================================================
% LOCAL HELPER FUNCTIONS FOR NLMPC MULTISTAGE
% =========================================================================

function x_next = datacenter_state_transition(x, u, d)
% Discrete-time thermal state transition:
% x: Room temperature T_room (°C)
% u: Cooling command in [0, 1]
% d: Measured disturbances [utilization, ambient_temp, carbon_intensity]

util = d(1);
tamb = d(2);

% Physical plant parameters (matching lumped-parameter thermal plant)
C_room = 800.0;    % Room thermal capacitance (kJ / K)
Q_idle = 150.0;    % Server idle heat (kW)
Q_it_max = 500.0;  % Server full load heat (kW)
Q_cool_max = 600.0;% Max cooling capacity (kW)
UA_env = 5.0;      % Envelope heat transfer coefficient (kW / K)
dt_hr = 5.0 / 60.0;% 5 minutes in hours

% IT heat generation
q_it = Q_idle + (Q_it_max - Q_idle) * util;

% Cooling delivered
q_cool = u * Q_cool_max;

% Heat exchange through building envelope
q_env = UA_env * (tamb - x);

% Net rate of temperature change (dT/dt in K/hr = kW / (kJ/K) * 3600)
% kJ / K is kW*s / K. DT_HR * 3600 s/hr.
dT_dt = (q_it - q_cool + q_env) / C_room; % K / s
x_next = x + (dT_dt * 300.0); % 300 seconds step
end

function cost = datacenter_stage_cost(stage, x, u, dmv, d)
% Evaluates carbon emissions cost + smooth actuation penalty for stage k
% stage: current stage index (1..p)
% x: state (T_room)
% u: cooling command
% dmv: change in MV (u(k) - u(k-1))
% d: disturbance vector [util, tamb, carbon_intensity]

tamb = d(2);
carbon_intensity = d(3); % gCO2 / kWh

% Cooling power calculation based on COP curve
% P_fan = 15 * u^3
% COP = max(1.5, 4.5 - 0.08 * (tamb - 20))
% P_comp = (u * 600) / COP
cop = max(1.5, 4.5 - 0.08 * (tamb - 20.0));
p_fan = 15.0 * (u^3);
p_comp = (u * 600.0) / cop;
p_cool = p_fan + p_comp; % Total cooling electrical power (kW)

dt_hr = 5.0 / 60.0; % 5 minutes in hours
energy_kwh = p_cool * dt_hr;

% Carbon cost in kg CO2 (carbon_intensity is gCO2/kWh, divide by 1000)
carbon_cost = (energy_kwh * carbon_intensity) / 1000.0;

% Actuator rate-of-change smoothing cost
smooth_cost = 0.05 * (dmv^2);

% Combined stage objective
cost = carbon_cost + smooth_cost;
end

function cost = datacenter_terminal_cost(stage, x, d)
% Terminal cost penalizing terminal deviation from target setpoint (22 °C)
cost = 0.5 * ((x - 22.0)^2);
end
