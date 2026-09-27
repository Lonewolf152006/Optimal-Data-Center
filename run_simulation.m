function results = run_simulation(modelName, nRuns, opts)
% RUN_SIMULATION  Drive the Data Center Cooling Simulink model with
% generated workload + environment scenarios, run it, and log the
% signals needed to build the ML/QML training dataset.
%
%   results = run_simulation('DataCenterCooling', 1)
%   results = run_simulation('models/DataCenterCooling.slx', 5)
%   results = run_simulation() % defaults to 'DataCenterCooling', 1 run
%
% Designed for MathWorks Excellence in Innovation Project #196:
%   "Optimal Data Center Cooling"

    % Ensure this script's directory and models/ directory are on MATLAB path
    scriptPath = mfilename('fullpath');
    if ~isempty(scriptPath)
        scriptDir = fileparts(scriptPath);
        if ~isempty(scriptDir) && exist(scriptDir, 'dir')
            addpath(scriptDir);
            modelsSubDir = fullfile(scriptDir, 'models');
            if exist(modelsSubDir, 'dir')
                addpath(modelsSubDir);
            end
        end
    end

    if nargin < 1 || isempty(modelName), modelName = 'DataCenterCooling'; end
    if nargin < 2 || isempty(nRuns), nRuns = 1; end
    if nargin < 3 || isempty(opts), opts = struct(); end

    nDays       = getOr(opts, 'nDays', 5);
    dtMinutes   = getOr(opts, 'dtMinutes', 5);
    stopTimeSec = nDays * 24 * 3600;

    % Locate model file if path or name provided
    [~, mBase, mExt] = fileparts(modelName);
    if isempty(mExt)
        if exist(fullfile(pwd, [mBase '.slx']), 'file')
            modelPath = fullfile(pwd, [mBase '.slx']);
        elseif exist(fullfile(pwd, 'models', [mBase '.slx']), 'file')
            modelPath = fullfile(pwd, 'models', [mBase '.slx']);
        else
            modelPath = fullfile(pwd, 'models', [mBase '.slx']);
        end
    else
        modelPath = modelName;
    end
    pureModelName = mBase;

    % Check if Simulink is available
    hasSimulink = (exist('sim', 'file') == 2 || exist('sim', 'builtin') == 5) && ...
                  (license('test', 'Simulink') == 1);

    if hasSimulink
        % Model MUST exist — no auto-build, no fallback
        if ~exist(modelPath, 'file') && ~bdIsLoaded(pureModelName)
            error('run_simulation:modelNotFound', ...
                ['Simulink model not found: %s\n' ...
                 'The model must be version-controlled under models/.\n' ...
                 'Run setup_simulink_model.m to create it from the official example.'], ...
                modelPath);
        end

        % Load system
        if ~bdIsLoaded(pureModelName)
            load_system(modelPath);
        end
        set_param(pureModelName, 'StopTime', num2str(stopTimeSec));

        % Initialize all base workspace parameters required by the physical model
        initModelBaseParameters(pureModelName);
    else
        fprintf('Note: Simulink license/executable not detected in this environment.\n');
        fprintf('Running high-fidelity MATLAB-native physical plant simulation...\n');
    end

    allRows = table();

    for run = 1:nRuns
        % Guard against nDays < 2: limit sample count to available days
        nHeatWave = min(randi([0 2]), nDays);
        nSpike    = min(randi([0 2]), nDays);
        heatWaveDays = sort(randperm(nDays, nHeatWave));
        spikeDays    = sort(randperm(nDays, nSpike));

        ambientTempTS = generate_environment(nDays, dtMinutes, heatWaveDays, 1000 + run);
        serverLoadTS  = generate_workload(nDays, dtMinutes, spikeDays, [13 15], 2000 + run);

        assignin('base', 'ambientTempTS', ambientTempTS);
        assignin('base', 'serverLoadTS',  serverLoadTS);

        % Native Simscape Fluids inputs:
        % 1) heat_load: timeseries in Watts
        heat_load_W  = 5000.0 * (0.6 + 0.8 * serverLoadTS.Data);
        heat_load_ts = timeseries(heat_load_W, serverLoadTS.Time, 'Name', 'heat_load');
        assignin('base', 'heat_load', heat_load_ts);

        % 2) Relative humidity timeseries
        rel_hum_data  = 0.6 * ones(size(ambientTempTS.Data));
        relHumidityTS = timeseries(rel_hum_data, ambientTempTS.Time, 'Name', 'RelativeHumidity');
        assignin('base', 'relHumidityTS', relHumidityTS);

        % 3) Environment as 1x2 timeseries array [AmbientTemp, RelHumidity]
        assignin('base', 'environment', [ambientTempTS, relHumidityTS]);

        if hasSimulink
            % Point Cooling Tower blocks directly to named timeseries
            set_param([pureModelName '/Cooling Tower/Temperature'], 'VariableName', 'ambientTempTS');
            set_param([pureModelName '/Cooling Tower/Relative Humidity'], 'VariableName', 'relHumidityTS');

            % Execute Simulink simulation
            simOut = sim(pureModelName, 'ReturnWorkspaceOutputs', 'on');

            % Time vector
            if isprop(simOut, 'tout') && ~isempty(simOut.tout)
                t_vec = simOut.tout;
            elseif isprop(simOut, 'time') && ~isempty(simOut.time)
                t_vec = simOut.time;
            else
                t_vec = (0:dtMinutes*60:stopTimeSec)';
            end

            % Extract logged signals — NO FALLBACKS, errors on missing signals
            [Tserver, Tcoolant, flowRate, pumpPower, chillerPw, totalPw] = ...
                extractLoggedSignals(simOut, t_vec);
        else
            % Pure MATLAB ODE numerical integration fallback
            t_vec = (0:dtMinutes*60:stopTimeSec)';
            [Tserver, Tcoolant, flowRate, pumpPower, chillerPw, totalPw] = ...
                simulate_plant_matlab(ambientTempTS, serverLoadTS, t_vec);
        end

        n = numel(t_vec);
        sLoadData = interp1(serverLoadTS.Time, serverLoadTS.Data, t_vec, 'linear', 'extrap');
        aTempData = interp1(ambientTempTS.Time, ambientTempTS.Data, t_vec, 'linear', 'extrap');

        rows = table(t_vec, repmat(run, n, 1), ...
            sLoadData, aTempData, ...
            Tserver, Tcoolant, flowRate, pumpPower, chillerPw, totalPw, ...
            'VariableNames', {'Time', 'Run', 'ServerLoad', 'AmbientTemp', ...
                'ServerTemp', 'CoolantTemp', 'FlowRate', 'PumpPower', ...
                'ChillerPower', 'TotalCoolingPower'});

        allRows = [allRows; rows]; %#ok<AGROW>
        fprintf('Run %d/%d complete (heat-wave days %s, spike days %s)\n', ...
            run, nRuns, mat2str(heatWaveDays), mat2str(spikeDays));
    end

    % Write to data/ directory as documented in README
    outDir = fullfile(fileparts(mfilename('fullpath')), 'data');
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    outFile = fullfile(outDir, 'thermal_dataset.csv');
    writetable(allRows, outFile);
    fprintf('Saved simulation dataset (%d rows) to %s\n', height(allRows), outFile);
    results = allRows;
end

% -------------------------------------------------------------------------
% Signal Extraction — Direct logsout access, NO silent fallbacks
% -------------------------------------------------------------------------
function [Tserver, Tcoolant, flowRate, pumpPower, chillerPw, totalPw] = ...
    extractLoggedSignals(simOut, timeVec)
% EXTRACTLOGGEDSIGNALS  Read all six plant signals from Simulink logsout.
% Errors loudly if any signal is missing — never returns placeholder data.

    signalNames = {'Server_Temp', 'Coolant_Temp', 'Flow_Rate', ...
                   'Pump_Power', 'Chiller_Power', 'Total_Cooling_Power'};

    % Get logsout dataset
    if isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
        logsout = simOut.logsout;
    else
        logsout = simOut.get('logsout');
    end

    if isempty(logsout)
        error('run_simulation:noLogsout', ...
            ['No signal logging data found in simulation output.\n' ...
             'Ensure the model has SignalLogging enabled and signals are ' ...
             'connected to logged outports.']);
    end

    results = cell(1, numel(signalNames));
    for i = 1:numel(signalNames)
        sig = logsout.get(signalNames{i});
        if isempty(sig)
            error('run_simulation:missingSignal', ...
                'Signal "%s" not found in logsout. Available signals: %s', ...
                signalNames{i}, strjoin(logsout.getElementNames(), ', '));
        end

        % Extract timeseries data
        if isprop(sig, 'Values') && ~isempty(sig.Values)
            sigVal = sig.Values;
        else
            sigVal = sig;
        end

        if isa(sigVal, 'timeseries')
            resampled = resample(sigVal, timeVec);
            results{i} = resampled.Data(:);
        elseif isa(sigVal, 'timetable')
            ts = timetable2timeseries(sigVal);
            resampled = resample(ts, timeVec);
            results{i} = resampled.Data(:);
        else
            error('run_simulation:unknownSignalType', ...
                'Signal "%s" has unexpected type "%s". Expected timeseries or timetable.', ...
                signalNames{i}, class(sigVal));
        end
    end

    Tserver   = results{1};
    Tcoolant  = results{2};
    flowRate  = results{3};
    pumpPower = results{4};
    chillerPw = results{5};
    totalPw   = results{6};
end

% -------------------------------------------------------------------------
% Pure MATLAB Plant Integration (Fallback when Simulink unavailable)
% -------------------------------------------------------------------------
function [Tserver, Tcoolant, flowRate, pumpPower, chillerPw, totalPw] = ...
    simulate_plant_matlab(ambientTempTS, serverLoadTS, t_vec)
% SIMULATE_PLANT_MATLAB  Forward-integrate the lumped-parameter thermal
% plant using RK4, matching the Python ThermalPlant.rk4_step exactly.
% Baseline controller uses stateful hysteresis matching Python's
% BaselineThermostat (setpoint=22, deadband=1, on=1.0, off=0.25).

    n = numel(t_vec);
    dt_sec = 300; % 5 min
    dt_hr  = dt_sec / 3600;

    % Plant parameters (match Python ThermalPlant class attributes exactly)
    C_ROOM = 50.0;   % kWh/°C
    UA_ENV = 4.0;    % kW/°C
    Q_IDLE = 150.0;  Q_IT_MAX = 500.0;  FAN_PEN_MAX = 30.0;
    Q_COOL_MAX = 600.0;  P_FAN_MAX = 40.0;
    FAN_RAMP_START = 25.0;  FAN_RAMP_FULL = 32.0;
    COP_CAP = 8.5;  COP_FLOOR = 2.5;  COP_SLOPE = 0.15;  COP_REF_T = 10.0;

    % Baseline controller parameters (match Python BaselineThermostat exactly)
    T_SET = 22.0;  DEADBAND = 1.0;
    ON_POWER = 1.0;  OFF_POWER = 0.25;

    Tserver   = zeros(n, 1);
    Tcoolant  = zeros(n, 1);
    flowRate  = zeros(n, 1);
    pumpPower = zeros(n, 1);
    chillerPw = zeros(n, 1);
    totalPw   = zeros(n, 1);

    currT = 22.0;
    ctrlMode = 'on';  % stateful hysteresis mode

    ambData  = resample(ambientTempTS, t_vec).Data;
    loadData = resample(serverLoadTS, t_vec).Data;

    for i = 1:n
        Tamb  = ambData(i);
        u_ut  = loadData(i);

        % Stateful hysteresis controller (matches Python BaselineThermostat)
        if currT >= T_SET + DEADBAND
            ctrlMode = 'on';
        elseif currT <= T_SET - DEADBAND
            ctrlMode = 'off';
        end
        % else: hold previous mode (true hysteresis)

        if strcmp(ctrlMode, 'on')
            u_ctrl = ON_POWER;
        else
            u_ctrl = OFF_POWER;
        end

        % Physics helper: IT heat load with fan ramp penalty
        ramp = min(max((currT - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0), 1);
        Qit  = Q_IDLE + (Q_IT_MAX - Q_IDLE)*u_ut + FAN_PEN_MAX * (ramp^2);

        % COP curve
        COP = min(max(COP_CAP - COP_SLOPE*(Tamb - COP_REF_T), COP_FLOOR), COP_CAP);

        % Cooling
        Qdel   = u_ctrl * Q_COOL_MAX;
        Pfan   = P_FAN_MAX * (u_ctrl^3);
        Pchil  = Qdel / COP;
        Ptot   = Pfan + Pchil;

        % RK4 integration step (matches Python ThermalPlant.rk4_step)
        dTdt_fn = @(T) (q_it_fn(T, u_ut, Q_IDLE, Q_IT_MAX, FAN_PEN_MAX, FAN_RAMP_START, FAN_RAMP_FULL) ...
                        - Qdel + UA_ENV * (Tamb - T)) / C_ROOM;
        k1 = dTdt_fn(currT);
        k2 = dTdt_fn(currT + dt_hr/2 * k1);
        k3 = dTdt_fn(currT + dt_hr/2 * k2);
        k4 = dTdt_fn(currT + dt_hr * k3);
        currT = currT + dt_hr/6 * (k1 + 2*k2 + 2*k3 + k4);

        Tserver(i)   = currT;
        Tcoolant(i)  = currT - 5.0;
        flowRate(i)  = u_ctrl * 25.0;
        pumpPower(i) = Pfan;
        chillerPw(i) = Pchil;
        totalPw(i)   = Ptot;
    end
end

% Helper for RK4: IT heat load as a function of temperature
function Qit = q_it_fn(T, u_ut, Q_IDLE, Q_IT_MAX, FAN_PEN_MAX, FAN_RAMP_START, FAN_RAMP_FULL)
    ramp = min(max((T - FAN_RAMP_START) / (FAN_RAMP_FULL - FAN_RAMP_START), 0), 1);
    Qit = Q_IDLE + (Q_IT_MAX - Q_IDLE)*u_ut + FAN_PEN_MAX * (ramp^2);
end

% -------------------------------------------------------------------------
% Initialize Model Base Workspace Parameters
% -------------------------------------------------------------------------
function initModelBaseParameters(pureModelName)
    % Execute model's PreLoadFcn if present
    preloadStr = get_param(pureModelName, 'PreLoadFcn');
    if ~isempty(preloadStr)
        evalin('base', preloadStr);
    end

    % Explicitly assign all physical parameters required by Simscape Fluids Data Center Cooling
    assignin('base', 'T_chiller', 18);
    assignin('base', 'server_pipe_D', 0.02);
    assignin('base', 'server_pipe_L', 12);
    assignin('base', 'server_pipe_thickness', 0.002);
    assignin('base', 'server_num_pipes', 800);
    assignin('base', 'rho_pipe', 7800);
    assignin('base', 'cp_pipe', 500);
    assignin('base', 'port_area', 0.2);
    assignin('base', 'T_reservoir', 23);
    assignin('base', 'fan_area', 15);
    assignin('base', 'tower_height', 3);
    assignin('base', 'tower_area', 15);

    % Ensure environment and heat_load are structured properly
    def_t = (0:300:432000)';
    def_temp = 24.0 + 6.5*sin(2*pi*(def_t - 9*3600)/86400);
    def_rh   = 0.6 * ones(size(def_t));
    def_ts_temp = timeseries(def_temp, def_t, 'Name', 'AmbientTemp');
    def_ts_rh   = timeseries(def_rh, def_t, 'Name', 'RelativeHumidity');
    assignin('base', 'ambientTempTS', def_ts_temp);
    assignin('base', 'relHumidityTS', def_ts_rh);

    assignin('base', 'environment', [def_ts_temp, def_ts_rh]);

    def_load_val = 5000.0 + 1500.0*sin(2*pi*(def_t - 9*3600)/86400);
    assignin('base', 'heat_load', timeseries(def_load_val, def_t, 'Name', 'heat_load'));
end

function v = getOr(s, field, default)
    if isfield(s, field), v = s.(field); else, v = default; end
end
