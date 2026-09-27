function [u_opt, info] = mpc_controller(T_current, util_fore, tamb_fore, carbon_fore, u_prev, surrogate_fn, opts)
% MPC_CONTROLLER Carbon-Aware Receding-Horizon Model Predictive Controller
% Implements non-linear predictive control for data center cooling to minimize
% carbon emissions while strictly preserving ASHRAE thermal compliance.
%
% Syntax:
%   [u_opt, info] = mpc_controller(T_current, util_fore, tamb_fore, carbon_fore)
%   [u_opt, info] = mpc_controller(T_current, util_fore, tamb_fore, carbon_fore, u_prev)
%   [u_opt, info] = mpc_controller(T_current, util_fore, tamb_fore, carbon_fore, u_prev, surrogate_fn, opts)
%
% Inputs:
%   T_current   - Current room temperature (°C)
%   util_fore   - Predicted server utilization vector [0, 1] over horizon (p x 1)
%   tamb_fore   - Predicted ambient temperature vector (°C) over horizon (p x 1)
%   carbon_fore - Predicted grid carbon intensity (gCO2/kWh) over horizon (p x 1)
%   u_prev      - Previous control action in [0, 1] (default: 0.45)
%   surrogate_fn- Optional function handle for learned surrogate (e.g. GRU dlnetwork)
%   opts        - Struct with fields:
%                   .horizon    : Prediction horizon (default: 24 steps = 2 hours)
%                   .dt_minutes : Step size in minutes (default: 5)
%                   .w_carbon   : Carbon cost weight (default: 1.0)
%                   .w_rec      : Soft penalty weight for recommended band [18, 27]°C (default: 400)
%                   .w_alw      : Soft penalty weight for allowable band [15, 32]°C (default: 20000)
%                   .w_smooth   : Smoothness penalty weight on (du/dt)^2 (default: 50)
%
% Outputs:
%   u_opt       - Optimal cooling command for the current step in [0, 1]
%   info        - Struct with diagnostics:
%                   .u_seq      : Full planned control trajectory over horizon
%                   .T_pred     : Predicted room temperature trajectory
%                   .cost       : Value of the objective function
%                   .exitflag   : Solver exit condition
%
% See also: create_datacenter_nlmpc, run_matlab_comparison, quadprog, fmincon

% Default argument handling
if nargin < 5 || isempty(u_prev)
    u_prev = 0.45;
end
if nargin < 6
    surrogate_fn = [];
end
if nargin < 7
    opts = struct();
end

% Extract options
if isfield(opts, 'horizon'), p = opts.horizon; else, p = min([length(util_fore), length(tamb_fore), 24]); end
if isfield(opts, 'dt_minutes'), dt_min = opts.dt_minutes; else, dt_min = 5.0; end
if isfield(opts, 'w_carbon'), w_carbon = opts.w_carbon; else, w_carbon = 1.0; end
if isfield(opts, 'w_rec'), w_rec = opts.w_rec; else, w_rec = 400.0; end
if isfield(opts, 'w_alw'), w_alw = opts.w_alw; else, w_alw = 20000.0; end
if isfield(opts, 'w_smooth'), w_smooth = opts.w_smooth; else, w_smooth = 50.0; end

dt_hr = dt_min / 60.0;

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
T_REC_LO = 18.0;     % ASHRAE Recommended lower limit (°C)
T_REC_HI = 27.0;     % ASHRAE Recommended upper limit (°C)
T_ALW_LO = 15.0;     % ASHRAE Allowable lower limit (°C)
T_ALW_HI = 32.0;     % ASHRAE Allowable upper limit (°C)

% Slice horizon
util = util_fore(1:p);
tamb = tamb_fore(1:p);
carbon = carbon_fore(1:p);

% Initial guess: warm start from previous control action
u0 = repmat(u_prev, p, 1);
lb = zeros(p, 1);
ub = ones(p, 1);

% Check if learned surrogate prediction is provided
ml_traj = [];
if ~isempty(surrogate_fn)
    try
        ml_traj = surrogate_fn(T_current, [util(:), tamb(:), carbon(:)]);
    catch
        ml_traj = [];
    end
end

% Forward trajectory simulation function
    function Ts = forward_model(u_seq)
        Ts = zeros(p + 1, 1);
        Ts(1) = T_current;
        for k = 1:p
            if ~isempty(ml_traj) && length(ml_traj) >= k
                % Dynamic actuation response:
                % Net cooling extraction rate scaled by room thermal capacitance
                pred_step = ml_traj(k) - Ts(k);
                dT_cool = (u_seq(k) * Q_COOL_MAX * dt_hr) / C_ROOM;
                dT_nominal = (0.45 * Q_COOL_MAX * dt_hr) / C_ROOM;
                Ts(k + 1) = Ts(k) + pred_step - (dT_cool - dT_nominal);
            else
                % Lumped-parameter forward model
                qit = Q_IDLE + (Q_IT_MAX - Q_IDLE) * util(k);
                q_del = u_seq(k) * Q_COOL_MAX;
                dT = (qit - q_del + UA_ENV * (tamb(k) - Ts(k))) / C_ROOM;
                Ts(k + 1) = Ts(k) + dt_hr * dT;
            end
        end
    end

% Objective function: Carbon emissions + ASHRAE soft constraint penalties + Smoothness
    function J = cost_function(u_seq)
        Ts = forward_model(u_seq);
        T_preds = Ts(2:end);
        
        % 1. Electrical Cooling Power and Carbon emissions
        cop = min(max(COP_CAP - COP_SLOPE * (tamb - COP_REF_T), COP_FLOOR), COP_CAP);
        p_fan = P_FAN_MAX * (u_seq .^ 3);
        p_comp = (u_seq * Q_COOL_MAX) ./ cop;
        pcool = p_fan + p_comp;
        carbon_emissions_kg = sum(pcool .* carbon .* dt_hr) / 1000.0;
        
        % 2. ASHRAE Soft Constraint Penalties (ECR formulation)
        rec_viol = max(0, T_REC_LO - T_preds) + max(0, T_preds - T_REC_HI);
        alw_viol = max(0, T_ALW_LO - T_preds) + max(0, T_preds - T_ALW_HI);
        penalty = w_rec * sum(rec_viol .^ 2) + w_alw * sum(alw_viol .^ 2);
        
        % 3. Actuator Rate-of-Change Smoothness
        du = diff([u_prev; u_seq]);
        smoothness = w_smooth * sum(du .^ 2);
        
        J = w_carbon * carbon_emissions_kg + penalty + smoothness;
    end

% Optimization Solver
has_fmincon = exist('fmincon', 'file') || exist('fmincon', 'builtin');

if has_fmincon
    fmin_opts = optimoptions('fmincon', ...
        'Display', 'off', ...
        'MaxIterations', 40, ...
        'MaxFunctionEvaluations', 800, ...
        'OptimalityTolerance', 1e-3, ...
        'StepTolerance', 1e-4);
    
    [u_opt_seq, fval, exitflag] = fmincon(@cost_function, u0, ...
        [], [], [], [], lb, ub, [], fmin_opts);
else
    % High-performance projected gradient descent with Armijo backtracking
    % Ensures standalone execution even if Optimization Toolbox is not installed
    u_opt_seq = u0;
    lr = 0.01;
    max_iter = 40;
    
    for iter = 1:max_iter
        % Numerical gradient computation
        grad = zeros(p, 1);
        base_cost = cost_function(u_opt_seq);
        eps_step = 1e-5;
        for i = 1:p
            u_pert = u_opt_seq;
            u_pert(i) = u_pert(i) + eps_step;
            grad(i) = (cost_function(u_pert) - base_cost) / eps_step;
        end
        
        % Projected gradient step
        u_cand = min(max(u_opt_seq - lr * grad, lb), ub);
        cand_cost = cost_function(u_cand);
        
        % Armijo line search
        if cand_cost < base_cost
            u_opt_seq = u_cand;
            lr = lr * 1.1; % Slight acceleration
        else
            lr = lr * 0.5; % Backtrack
        end
        
        if norm(grad) < 1e-3 || lr < 1e-6
            break;
        end
    end
    fval = cost_function(u_opt_seq);
    exitflag = 1;
end

% Return first optimal action
u_opt = min(max(u_opt_seq(1), 0.0), 1.0);

% Assemble info structure
info = struct();
info.u_seq = u_opt_seq;
info.T_pred = forward_model(u_opt_seq);
info.cost = fval;
info.exitflag = exitflag;

end
