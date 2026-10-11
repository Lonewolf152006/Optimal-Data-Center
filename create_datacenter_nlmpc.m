function nlobj = create_datacenter_nlmpc(p)
% CREATE_DATACENTER_NLMPC Configures a multistage nonlinear MPC controller
% for optimal, carbon-aware data center cooling using Model Predictive Control Toolbox.
%
% Fixes MathWorks reviewer item #5:
%   - Physical plant parameters match Python ThermalPlant and MATLAB plant exactly:
%     C_ROOM = 50.0 kWh/°C, UA_ENV = 4.0 kW/°C, Q_COOL_MAX = 600 kW, P_FAN_MAX = 40 kW
%   - Exact COP degradation curve: COP = clip(8.5 - 0.15*(Tamb - 10), 2.5, 8.5)
%   - Server fan ramp penalty included in state transition
%   - ASHRAE Recommended [18, 27] °C and Allowable [15, 32] °C bounds configured
%     as soft constraints with Equal Concern for Relaxation (ECR) weights
%   - Stage cost evaluates grid carbon emissions: P_cool(u, Tamb) * CarbonIntensity * dt
%
% Syntax:
%   nlobj = create_datacenter_nlmpc()
%   nlobj = create_datacenter_nlmpc(p)
%
% Inputs:
%   p - Prediction horizon in steps (default: 24 steps = 2 hours at dt = 5 min)
%
% Outputs:
%   nlobj - Configured nlmpcMultistage object (or configuration struct if toolbox absent)
%
% See also: nlmpcMultistage, mpc_controller, run_matlab_comparison

if nargin < 1 || isempty(p)
    p = 24; % 24 steps * 5 minutes = 2 hours prediction horizon
end

nx = 1; % State: Room Temperature [deg C]
nu = 1; % MV: Cooling actuation command u in [0, 1]
nd = 3; % Measured Disturbances: [utilization, ambient_temp, carbon_intensity]

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
    nlobj.C_ROOM = 50.0;
    nlobj.UA_ENV = 4.0;
    nlobj.Q_COOL_MAX = 600.0;
    return;
end

% Create Multistage Nonlinear MPC object
nlobj = nlmpcMultistage(p, nx, nu);
nlobj.Ts = 300; % 5 minutes sample time (seconds)

% Manipulated Variable bounds & rate constraints
nlobj.MV.Min = 0.0;
nlobj.MV.Max = 1.0;
nlobj.MV.RateMin = -0.4; % Rate limiter for chiller/fan longevity
nlobj.MV.RateMax = 0.4;

% Specify discrete-time state transition function: x(k+1) = f(x(k), u(k), d(k))
nlobj.Model.StateFcn = @datacenter_state_transition;
nlobj.Model.IsContinuousTime = false;

% State bounds: ASHRAE Allowable Envelope [15, 32] °C with soft ECR weights
nlobj.States(1).Min = 15.0;
nlobj.States(1).Max = 32.0;

% Equal Concern for Relaxation (ECR) weights for soft boundary enforcement
if isprop(nlobj.States(1), 'MinECR')
    nlobj.States(1).MinECR = 1.0;
    nlobj.States(1).MaxECR = 1.0;
end

% Assign Stage Cost for all prediction stages (carbon footprint + recommended band ECR)
for k = 1:p
    nlobj.Stages(k).CostFcn = @datacenter_stage_cost;
end

% Terminal stage cost
nlobj.Stages(p+1).CostFcn = @datacenter_terminal_cost;

end

% =========================================================================
% LOCAL HELPER FUNCTIONS FOR NLMPC MULTISTAGE
% =========================================================================

function x_next = datacenter_state_transition(x, u, d)
% Discrete-time thermal state transition matching ThermalPlant physics:
%   x: Room temperature T_room (°C)
%   u: Cooling command in [0, 1]
%   d: Measured disturbances [utilization, ambient_temp, carbon_intensity]

util = min(max(d(1), 0.0), 1.0);
tamb = d(2);
u_ctrl = min(max(u, 0.0), 1.0);

% Physical plant parameters (MUST match ThermalPlant class attributes)
C_ROOM   = 50.0;       % kWh/°C
UA_ENV   = 4.0;        % kW/°C
Q_IDLE   = 150.0;      % kW
Q_IT_MAX = 500.0;      % kW
FAN_PEN_MAX    = 30.0; % kW
FAN_RAMP_START = 25.0; % °C
FAN_RAMP_FULL  = 32.0; % °C
Q_COOL_MAX     = 600.0;% kW
dt_hr          = 5.0 / 60.0; % 5 minutes = 0.0833 hr

% IT heat generation with fan ramp penalty
ramp = min(max((x - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0.0), 1.0);
q_it = Q_IDLE + (Q_IT_MAX - Q_IDLE) * util + FAN_PEN_MAX * (ramp^2);

% Cooling heat extraction
q_del = u_ctrl * Q_COOL_MAX;

% Envelope heat transfer
q_env = UA_ENV * (tamb - x);

% State advance: discrete 1-step forward Euler / midpoint approximation for MPC
dT = (q_it - q_del + q_env) / C_ROOM;
x_next = x + dt_hr * dT;
end

function cost = datacenter_stage_cost(stage, x, u, dmv, d)
% Evaluates carbon emissions cost + soft ASHRAE recommended band violation
%   stage: prediction stage index (1..p)
%   x: state (T_room in °C)
%   u: cooling command in [0, 1]
%   dmv: rate of change (u(k) - u(k-1))
%   d: disturbance vector [utilization, ambient_temp, carbon_intensity]

tamb = d(2);
carbon_intensity = d(3); % gCO2 / kWh

% Chiller COP curve
COP_CAP   = 8.5;
COP_FLOOR = 2.5;
COP_SLOPE = 0.15;
COP_REF_T = 10.0;
cop = min(max(COP_CAP - COP_SLOPE * (tamb - COP_REF_T), COP_FLOOR), COP_CAP);

% Electrical power consumption (kW)
P_FAN_MAX  = 40.0;
Q_COOL_MAX = 600.0;
p_fan = P_FAN_MAX * (u^3);
p_comp = (u * Q_COOL_MAX) / cop;
p_cool = p_fan + p_comp;

dt_hr = 5.0 / 60.0;
energy_kwh = p_cool * dt_hr;

% 1. Stage carbon emissions (kg CO2)
carbon_cost = (energy_kwh * carbon_intensity) / 1000.0;

% 2. Soft ASHRAE Recommended Band [18, 27] °C penalty with ECR formulation
T_REC_LO = 18.0;
T_REC_HI = 27.0;
rec_viol = max(0.0, T_REC_LO - x) + max(0.0, x - T_REC_HI);
w_ecr = 10.0; % Soft ECR relaxation weight for recommended envelope
ecr_penalty = w_ecr * (rec_viol^2);

% 3. Smoothness penalty on actuator slew rate
smooth_cost = 0.05 * (dmv^2);

cost = carbon_cost + ecr_penalty + smooth_cost;
end

function cost = datacenter_terminal_cost(stage, x, d)
% Terminal cost: Soft penalty if terminal state is outside recommended band
T_REC_LO = 18.0;
T_REC_HI = 27.0;
rec_viol = max(0.0, T_REC_LO - x) + max(0.0, x - T_REC_HI);
cost = 25.0 * (rec_viol^2);
end
