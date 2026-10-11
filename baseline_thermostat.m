function [u, mode] = baseline_thermostat(T_room, reset)
% BASELINE_THERMOSTAT Single source-of-truth Relay thermostat with true hysteresis.
%
% Implements the definitive thermostatic baseline controller across all MATLAB
% scripts and Simulink models for MathWorks Challenge Project #196.
%
% Control Logic (Relay with Hysteresis):
%   - Setpoint: 22.0 °C, Deadband: ±1.0 °C
%   - Switch ON point:  T >= 23.0 °C  --> u = 1.00 (maximum cooling)
%   - Switch OFF point: T <= 21.0 °C  --> u = 0.25 (minimum cooling)
%   - Between 21.0 °C and 23.0 °C: Hold previous state (true memory hysteresis)
%
% Syntax:
%   u = baseline_thermostat(T_room)
%   [u, mode] = baseline_thermostat(T_room)
%   [u, mode] = baseline_thermostat(T_room, reset)  % pass reset=true to initialize
%   params = baseline_thermostat('params')
%
% Outputs:
%   u    - Cooling command in [0, 1] (1.0 for on, 0.25 for off)
%   mode - Current boolean state (true = 'on', false = 'off')
%
% See also: run_matlab_comparison, run_simulation, validate_plant_comparison

    % Parameter query mode
    if ischar(T_room) || isstring(T_room)
        if strcmpi(T_room, 'params')
            u = struct('setpoint', 22.0, 'deadband', 1.0, ...
                       'switch_on', 23.0, 'switch_off', 21.0, ...
                       'on_power', 1.00, 'off_power', 0.25);
            mode = 'on';
            return;
        end
    end

    persistent thermo_state;
    if isempty(thermo_state)
        thermo_state = true; % default initial state: 'on' (consistent with Python baseline)
    end

    if nargin >= 2 && reset
        thermo_state = true;
    end

    T_SET      = 22.0;
    DEADBAND   = 1.0;
    ON_POWER   = 1.00;
    OFF_POWER  = 0.25;

    T_SWITCH_ON  = T_SET + DEADBAND; % 23.0 °C
    T_SWITCH_OFF = T_SET - DEADBAND; % 21.0 °C

    if T_room >= T_SWITCH_ON
        thermo_state = true;
    elseif T_room <= T_SWITCH_OFF
        thermo_state = false;
    end
    % else: preserve previous thermo_state (true hysteresis)

    if thermo_state
        u = ON_POWER;
    else
        u = OFF_POWER;
    end

    mode = thermo_state;
end
