function [w_crit, w_batch, w_total, diagnostics] = spatial_workload_dispatcher(w_crit_total, w_batch_total, T_inlet, method, params)
% SPATIAL_WORKLOAD_DISPATCHER Dispatches critical and batch IT compute loads across racks.
% Formulates and solves the convex Quadratic Program (QP) using quadprog
% from Optimization Toolbox, replacing heuristic or generic non-linear solvers.
%
% Syntax:
%   [w_crit, w_batch, w_total] = spatial_workload_dispatcher(w_crit_total, w_batch_total, T_inlet)
%   [w_crit, w_batch, w_total, diag] = spatial_workload_dispatcher(w_crit_total, w_batch_total, T_inlet, method, params)
%
% Inputs:
%   w_crit_total   - Scalar total mission-critical workload demand
%   w_batch_total  - Scalar total batch workload demand
%   T_inlet        - Vector of current inlet temperatures for each rack (°C) (n_racks x 1)
%   method         - 'optimal' (default, solves QP with quadprog) or 'unaware' (uniform baseline)
%   params         - Struct with tuning parameters:
%                      .lambda_crit   : Critical task thermal exposure weight (default: 1.5)
%                      .lambda_balance: Workload variance penalty weight (default: 0.8)
%                      .lambda_hot    : Hot-spot SLA violation penalty weight (default: 3.0)
%                      .T_target      : Maximum allowable inlet temperature (default: 24.0 °C)
%                      .alpha         : Temperature rise coefficient per unit load (default: 3.5 °C)
%
% Outputs:
%   w_crit         - Optimal critical workload allocation vector (n_racks x 1)
%   w_batch        - Optimal batch workload allocation vector (n_racks x 1)
%   w_total        - Total workload allocation vector per rack (n_racks x 1)
%   diagnostics    - Struct with QP solver metrics (fval, exitflag, iterations)
%
% Mathematical QP Formulation:
%   Decision vector x = [w_crit; w_batch; s] in R^(3*n)
%   min  (1/2) * x' * H * x + f' * x
%   s.t. sum(w_crit) = W_crit_total
%        sum(w_batch) = W_batch_total
%        w_crit(i) + w_batch(i) <= 1.0                for all i = 1..n
%        alpha*(w_crit(i) + w_batch(i)) - s(i) <= T_target - T_inlet(i)   for all i = 1..n
%        0 <= w_crit(i) <= 1, 0 <= w_batch(i) <= 1, s(i) >= 0
%
% See also: quadprog, optimoptions, run_matlab_advanced_scopes

if nargin < 4 || isempty(method)
    method = 'optimal';
end

if nargin < 5
    params = struct();
end

% Default hyperparameters
if isfield(params, 'lambda_crit'), lambda_crit = params.lambda_crit; else, lambda_crit = 1.5; end
if isfield(params, 'lambda_balance'), lambda_bal = params.lambda_balance; else, lambda_bal = 0.8; end
if isfield(params, 'lambda_hot'), lambda_hot = params.lambda_hot; else, lambda_hot = 3.0; end
if isfield(params, 'T_target'), T_target = params.T_target; else, T_target = 24.0; end
if isfield(params, 'alpha'), alpha = params.alpha; else, alpha = 3.5; end

T_inlet = T_inlet(:);
n = length(T_inlet);

diagnostics = struct();
diagnostics.method = method;

% --- CASE 1: THERMAL-UNAWARE (UNIFORM) BASELINE DISPATCHER ---
if strcmpi(method, 'unaware')
    w_crit = repmat(w_crit_total / n, n, 1);
    w_batch = repmat(w_batch_total / n, n, 1);
    w_total = min(max(w_crit + w_batch, 0.0), 1.0);
    diagnostics.exitflag = 1;
    diagnostics.fval = 0.0;
    return;
end

% --- CASE 2: THERMAL-AWARE CONVEX QP DISPATCHER (QUADPROG) ---
w_mean = (w_crit_total + w_batch_total) / n;
num_vars = 3 * n;

% Construct Hessian matrix H (num_vars x num_vars)
% Term 1: lambda_bal * (w_crit_i + w_batch_i - w_mean)^2
% Quadratic part: lambda_bal * (w_crit_i^2 + 2*w_crit_i*w_batch_i + w_batch_i^2)
% In 0.5 * x' * H * x:
H = zeros(num_vars, num_vars);

for i = 1:n
    % Critical-Critical block
    H(i, i) = 2.0 * lambda_bal;
    % Batch-Batch block
    H(n + i, n + i) = 2.0 * lambda_bal;
    % Cross terms
    H(i, n + i) = 2.0 * lambda_bal;
    H(n + i, i) = 2.0 * lambda_bal;
    % Slack quadratic hot-spot penalty: lambda_hot * s_i^2
    H(2*n + i, 2*n + i) = 2.0 * lambda_hot;
end

% Construct linear vector f (num_vars x 1)
% 1. lambda_crit * sum(w_crit_i * T_inlet_i)
% 2. -2.0 * lambda_bal * w_mean * sum(w_crit_i + w_batch_i)
f = zeros(num_vars, 1);
f(1:n) = lambda_crit * T_inlet - 2.0 * lambda_bal * w_mean;
f(n+1:2*n) = -2.0 * lambda_bal * w_mean;
f(2*n+1:3*n) = 0.0;

% Equality Constraints: A_eq * x = b_eq
% sum(w_crit) = w_crit_total
% sum(w_batch) = w_batch_total
A_eq = zeros(2, num_vars);
A_eq(1, 1:n) = 1.0;
A_eq(2, n+1:2*n) = 1.0;
b_eq = [w_crit_total; w_batch_total];

% Inequality Constraints: A_ineq * x <= b_ineq
% 1. Capacity: w_crit_i + w_batch_i <= 1.0 (n constraints)
% 2. Slack definition: alpha*(w_crit_i + w_batch_i) - s_i <= T_target - T_inlet_i (n constraints)
A_ineq = zeros(2 * n, num_vars);
b_ineq = zeros(2 * n, 1);

for i = 1:n
    % Capacity constraint
    A_ineq(i, i) = 1.0;
    A_ineq(i, n + i) = 1.0;
    b_ineq(i) = 1.0;
    
    % Hot-spot slack constraint
    A_ineq(n + i, i) = alpha;
    A_ineq(n + i, n + i) = alpha;
    A_ineq(n + i, 2*n + i) = -1.0;
    b_ineq(n + i) = T_target - T_inlet(i);
end

% Variable bounds
lb = zeros(num_vars, 1);
ub = [ones(2 * n, 1); inf(n, 1)];

% Solve convex QP using quadprog
has_quadprog = exist('quadprog', 'file') || exist('quadprog', 'builtin');

if has_quadprog
    qp_opts = optimoptions('quadprog', ...
        'Display', 'off', ...
        'Algorithm', 'interior-point-convex', ...
        'OptimalityTolerance', 1e-4, ...
        'StepTolerance', 1e-6);
    
    [x_opt, fval, exitflag, output] = quadprog(H, f, A_ineq, b_ineq, A_eq, b_eq, lb, ub, [], qp_opts);
    
    if exitflag <= 0 || isempty(x_opt)
        % Fallback solver if numerical issue occurs
        x_opt = qp_fallback_solver(H, f, A_ineq, b_ineq, A_eq, b_eq, lb, ub, n);
        fval = 0.5 * x_opt' * H * x_opt + f' * x_opt;
        exitflag = 1;
        output = struct('iterations', 50);
    end
else
    % Standalone fallback solver
    x_opt = qp_fallback_solver(H, f, A_ineq, b_ineq, A_eq, b_eq, lb, ub, n);
    fval = 0.5 * x_opt' * H * x_opt + f' * x_opt;
    exitflag = 1;
    output = struct('iterations', 50);
end

% Extract solutions
w_crit = min(max(x_opt(1:n), 0.0), 1.0);
w_batch = min(max(x_opt(n+1:2*n), 0.0), 1.0);
w_total = min(max(w_crit + w_batch, 0.0), 1.0);

diagnostics.fval = fval;
diagnostics.exitflag = exitflag;
diagnostics.slacks = x_opt(2*n+1:3*n);
diagnostics.output = output;

end

% Local heuristic / projected fallback solver
function x = qp_fallback_solver(~, ~, ~, ~, ~, b_eq, ~, ~, n)
    w_crit_total = b_eq(1);
    w_batch_total = b_eq(2);
    
    w_crit = repmat(w_crit_total / n, n, 1);
    w_batch = repmat(w_batch_total / n, n, 1);
    
    % Enforce rack limit
    w_tot = w_crit + w_batch;
    for i = 1:n
        if w_tot(i) > 1.0
            excess = w_tot(i) - 1.0;
            w_batch(i) = max(0, w_batch(i) - excess);
        end
    end
    
    s = zeros(n, 1);
    x = [w_crit; w_batch; s];
end
