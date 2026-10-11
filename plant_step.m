function [T_next, diag] = plant_step(T_current, u_ctrl, util, Tamb, dt_hr)
% PLANT_STEP Single source-of-truth 4th-order Runge-Kutta simulation step for
% the lumped-parameter data center thermal plant.
%
% Exactly matches the physics of Python ThermalPlant.rk4_step down to machine precision.
%
% Syntax:
%   T_next = plant_step(T_current, u_ctrl, util, Tamb)
%   T_next = plant_step(T_current, u_ctrl, util, Tamb, dt_hr)
%   [T_next, diag] = plant_step(...)
%
% Inputs:
%   T_current - Current room air temperature (°C)
%   u_ctrl    - Actuation cooling command in [0, 1]
%   util      - Server IT utilization fraction in [0, 1]
%   Tamb      - Ambient air temperature (°C)
%   dt_hr     - Integration step size in hours (default: 5/60 = 0.0833 hr)
%
% Outputs:
%   T_next    - Temperature at next step (°C)
%   diag      - Struct with instantaneous power, heat, and COP diagnostics
%
% See also: baseline_thermostat, mpc_controller, run_simulation, validate_plant_comparison

    if nargin < 5 || isempty(dt_hr)
        dt_hr = 5.0 / 60.0; % 5 minutes in hours
    end

    % Physical constants (MUST match Python ThermalPlant class attributes)
    C_ROOM         = 50.0;       % Room thermal capacitance (kWh/°C)
    UA_ENV         = 4.0;        % Envelope heat transfer coefficient (kW/°C)
    Q_IDLE         = 150.0;      % Server idle heat (kW)
    Q_IT_MAX       = 500.0;      % Server full-load IT heat (kW)
    FAN_PEN_MAX    = 30.0;       % Max server fan ramp penalty (kW)
    FAN_RAMP_START = 25.0;       % Fan ramp onset (°C)
    FAN_RAMP_FULL  = 32.0;       % Fan ramp saturation (°C)
    Q_COOL_MAX     = 600.0;      % Chiller max cooling capacity (kW)
    P_FAN_MAX      = 40.0;       % Max CRAC fan electrical power (kW)
    COP_CAP        = 8.5;        % Chiller COP ceiling
    COP_FLOOR      = 2.5;        % Chiller COP floor
    COP_SLOPE      = 0.15;       % COP temperature sensitivity
    COP_REF_T      = 10.0;       % Reference ambient temperature (°C)

    % Clamp inputs to physical ranges
    u = min(max(u_ctrl, 0.0), 1.0);
    ut = min(max(util, 0.0), 1.0);

    % Cooling heat extraction
    q_del = u * Q_COOL_MAX;

    % Chiller COP and electrical power
    cop = min(max(COP_CAP - COP_SLOPE * (Tamb - COP_REF_T), COP_FLOOR), COP_CAP);
    p_fan = P_FAN_MAX * (u^3);
    p_comp = q_del / cop;
    p_cool = p_fan + p_comp;

    % Continuous derivative function: dT/dt = (Q_it - Q_cool + Q_env) / C_room
    function dT = deriv(T)
        ramp = min(max((T - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0.0), 1.0);
        qit = Q_IDLE + (Q_IT_MAX - Q_IDLE) * ut + FAN_PEN_MAX * (ramp^2);
        qenv = UA_ENV * (Tamb - T);
        dT = (qit - q_del + qenv) / C_ROOM;
    end

    % 4th-Order Runge-Kutta numerical integration
    k1 = deriv(T_current);
    k2 = deriv(T_current + 0.5 * dt_hr * k1);
    k3 = deriv(T_current + 0.5 * dt_hr * k2);
    k4 = deriv(T_current + dt_hr * k3);

    T_next = T_current + (dt_hr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    if nargout > 1
        ramp_now = min(max((T_current - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0.0), 1.0);
        diag = struct();
        diag.q_it = Q_IDLE + (Q_IT_MAX - Q_IDLE) * ut + FAN_PEN_MAX * (ramp_now^2);
        diag.q_del = q_del;
        diag.cop = cop;
        diag.p_cool = p_cool;
        diag.p_fan = p_fan;
        diag.p_comp = p_comp;
    end
end
