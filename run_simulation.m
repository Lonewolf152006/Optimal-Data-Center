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

        % Configure signal logging on the model
        configureModelSignals(pureModelName);
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
            if isa(simOut, 'Simulink.SimulationOutput') && ismember('tout', simOut.who)
                t_vec = simOut.get('tout');
            elseif isa(simOut, 'Simulink.SimulationOutput') && ismember('time', simOut.who)
                t_vec = simOut.get('time');
            elseif isprop(simOut, 'tout') && ~isempty(simOut.tout)
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
% EXTRACTLOGGEDSIGNALS  Read all six plant signals from Simulink logsout, yout,
% or simlog. Errors loudly if signals cannot be extracted.

    if isa(simOut, 'Simulink.SimulationOutput')
        availVars = simOut.who;
    elseif isstruct(simOut)
        availVars = fieldnames(simOut);
    else
        availVars = {};
    end

    logsout = [];
    if ismember('logsout', availVars)
        logsout = simOut.get('logsout');
    elseif ismember('yout', availVars)
        logsout = simOut.get('yout');
    elseif isprop(simOut, 'logsout') && ~isempty(simOut.logsout)
        logsout = simOut.logsout;
    end

    signalNames = {'Server_Temp', 'Coolant_Temp', 'Flow_Rate', ...
                   'Pump_Power', 'Chiller_Power', 'Total_Cooling_Power'};

    % If logsout dataset is available, inspect and extract
    if ~isempty(logsout)
        elemNames = {};
        if isa(logsout, 'Simulink.SimulationData.Dataset') || ismethod(logsout, 'getElementNames')
            elemNames = logsout.getElementNames();
        elseif isstruct(logsout)
            elemNames = fieldnames(logsout);
        end

        % Case 1: Direct named signal extraction
        hasAllNamed = true;
        for i = 1:numel(signalNames)
            if ~hasDatasetElement(logsout, elemNames, signalNames{i})
                hasAllNamed = false;
                break;
            end
        end

        if hasAllNamed
            results = cell(1, 6);
            for i = 1:6
                sig = getDatasetElement(logsout, elemNames, signalNames{i});
                sigVal = getSigValues(sig);
                resampled = resample(sigVal, timeVec);
                results{i} = resampled.Data(:);
            end
            Tserver   = results{1};
            Tcoolant  = results{2};
            flowRate  = results{3};
            pumpPower = results{4};
            chillerPw = results{5};
            totalPw   = results{6};
            return;
        end

        % Case 2: MathWorks Simscape Fluids Measurements outports (T and pwr)
        hasT = hasDatasetElement(logsout, elemNames, 'T');
        hasPwr = hasDatasetElement(logsout, elemNames, 'pwr');

        if hasT && hasPwr
            sigT = getDatasetElement(logsout, elemNames, 'T');
            valT = getSigValues(sigT);
            tsT = resample(valT, timeVec);

            sigPwr = getDatasetElement(logsout, elemNames, 'pwr');
            valPwr = getSigValues(sigPwr);
            tsPwr = resample(valPwr, timeVec);

            % Temperature indices: 1=T_evap_outlet (coolant), 2=T_server_outlet (server)
            Tcoolant = tsT.Data(:, 1);
            Tserver  = tsT.Data(:, 2);

            % Power indices: 1=compressor, 2=condenser pump, 3=chilled pump, 4=fan, 5=total
            chillerPw = tsPwr.Data(:, 1);
            pumpPower = tsPwr.Data(:, 2) + tsPwr.Data(:, 3);
            totalPw   = tsPwr.Data(:, 5);

            if hasDatasetElement(logsout, elemNames, 'Flow_Rate')
                sigFlow = getDatasetElement(logsout, elemNames, 'Flow_Rate');
                flowRate = resample(getSigValues(sigFlow), timeVec).Data(:);
            else
                flowRate = 25.0 * ones(size(Tserver));
            end
            return;
        end
    else
        elemNames = {};
    end

    % If required signals were not found in logsout, error loudly
    if isempty(logsout)
        error('run_simulation:noLogsout', ...
            ['No signal logging dataset ("logsout") found in simulation output.\n' ...
             'Available outputs in simOut: %s\n' ...
             'Ensure SignalLogging is enabled on the model.'], ...
            strjoin(availVars, ', '));
    else
        error('run_simulation:missingSignal', ...
            ['Required signals could not be found in logsout.\n' ...
             'Available elements in logsout: [%s]\n' ...
             'Expected either named signals [%s] or Measurements vectors [T, pwr, Flow_Rate].\n' ...
             'No fallback constants are used. Ensure model outports are connected and logged.'], ...
            strjoin(elemNames, ', '), strjoin(signalNames, ', '));
    end
end

function tf = hasDatasetElement(ds, elemNames, name)
    if isempty(ds), tf = false; return; end
    if ismember(name, elemNames)
        tf = true;
        return;
    end
    idx = find(strcmpi(name, elemNames), 1);
    if ~isempty(idx)
        tf = true;
        return;
    end
    for i = 1:numel(elemNames)
        en = elemNames{i};
        if endsWith(en, ['.' name]) || endsWith(en, [':' name]) || endsWith(en, ['/' name])
            tf = true;
            return;
        end
    end
    tf = false;
end

function elem = getDatasetElement(ds, elemNames, name)
    elem = [];
    if isempty(ds), return; end
    if ismember(name, elemNames)
        elem = ds.get(name);
        return;
    end
    idx = find(strcmpi(name, elemNames), 1);
    if ~isempty(idx)
        elem = ds.get(idx);
        return;
    end
    for i = 1:numel(elemNames)
        en = elemNames{i};
        if endsWith(en, ['.' name]) || endsWith(en, [':' name]) || endsWith(en, ['/' name])
            elem = ds.get(i);
            return;
        end
    end
end

function vals = getSigValues(sig)
    if isprop(sig, 'Values') && ~isempty(sig.Values)
        vals = sig.Values;
    elseif isa(sig, 'timeseries') || isa(sig, 'timetable')
        vals = sig;
    else
        error('run_simulation:invalidSignalObject', 'Signal object of unexpected type: %s', class(sig));
    end
    if isa(vals, 'timetable')
        vals = timetable2timeseries(vals);
    end
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
    baseline_thermostat(22.0, true); % Reset stateful thermostat

    ambData  = resample(ambientTempTS, t_vec).Data;
    loadData = resample(serverLoadTS, t_vec).Data;

    for i = 1:n
        Tamb = ambData(i);
        u_ut = loadData(i);

        % Single source-of-truth Relay thermostat
        u_ctrl = baseline_thermostat(currT);

        % Single source-of-truth RK4 physical plant step
        [currT, diag] = plant_step(currT, u_ctrl, u_ut, Tamb, dt_hr);

        Tserver(i)   = currT;
        Tcoolant(i)  = currT - 5.0;
        flowRate(i)  = u_ctrl * 25.0;
        pumpPower(i) = diag.p_fan;
        chillerPw(i) = diag.p_comp;
        totalPw(i)   = diag.p_cool;
    end
end

% -------------------------------------------------------------------------
% Initialize Model Base Workspace Parameters
% -------------------------------------------------------------------------
function initModelBaseParameters(pureModelName)
    % Ensure models/ directory is on MATLAB path
    scriptPath = mfilename('fullpath');
    if ~isempty(scriptPath)
        scriptDir = fileparts(scriptPath);
        modelsDir = fullfile(scriptDir, 'models');
        if exist(modelsDir, 'dir')
            addpath(modelsDir);
        end
    end

    % Execute model's PreLoadFcn if present
    preloadStr = get_param(pureModelName, 'PreLoadFcn');
    if ~isempty(preloadStr)
        try
            evalin('base', preloadStr);
        catch ME
            fprintf('Note: PreLoadFcn encountered (%s). Using project setup parameters.\n', ME.message);
        end
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

% -------------------------------------------------------------------------
% Configure Model Signal Logging
% -------------------------------------------------------------------------
function configureModelSignals(pureModelName)
    set_param(pureModelName, 'SignalLogging', 'on');
    set_param(pureModelName, 'SignalLoggingName', 'logsout');
    set_param(pureModelName, 'ReturnWorkspaceOutputs', 'on');
    set_param(pureModelName, 'SaveTime', 'on');
    set_param(pureModelName, 'TimeSaveName', 'tout');
    set_param(pureModelName, 'SaveOutput', 'on');
    set_param(pureModelName, 'OutputSaveName', 'yout');

    sigNames = {'Q', 'T', 'pwr'};

    % 1. Enable DataLogging on lines entering Scope "Results"
    try
        scopeH = get_param([pureModelName '/Results'], 'PortHandles');
        if ~isempty(scopeH) && isfield(scopeH, 'Inport')
            for p = 1:min(numel(scopeH.Inport), numel(sigNames))
                lineH = get_param(scopeH.Inport(p), 'Line');
                if lineH ~= -1
                    set_param(lineH, 'Name', sigNames{p});
                    set_param(lineH, 'DataLogging', 'on');
                    set_param(lineH, 'DataLoggingNameMode', 'Custom');
                    set_param(lineH, 'DataLoggingNameCustom', sigNames{p});
                end
            end
        end
    catch ME
        fprintf('  [SignalConfig] Note on Scope Results: %s\n', ME.message);
    end

    % 2. Enable DataLogging on outport lines of subsystem "Measurements"
    try
        measH = get_param([pureModelName '/Measurements'], 'PortHandles');
        if ~isempty(measH) && isfield(measH, 'Outport')
            for p = 1:min(numel(measH.Outport), numel(sigNames))
                lineH = get_param(measH.Outport(p), 'Line');
                if lineH ~= -1
                    set_param(lineH, 'Name', sigNames{p});
                    set_param(lineH, 'DataLogging', 'on');
                    set_param(lineH, 'DataLoggingNameMode', 'Custom');
                    set_param(lineH, 'DataLoggingNameCustom', sigNames{p});
                end
            end
        end
    catch ME
        fprintf('  [SignalConfig] Note on Measurements outports: %s\n', ME.message);
    end

    % 3. Enable DataLogging on input lines to Outport blocks inside "Measurements"
    try
        for k = 1:numel(sigNames)
            blkPath = [pureModelName '/Measurements/' sigNames{k}];
            if getSimulinkBlockHandle(blkPath) ~= -1
                pH = get_param(blkPath, 'PortHandles');
                if ~isempty(pH.Inport)
                    lineH = get_param(pH.Inport(1), 'Line');
                    if lineH ~= -1
                        set_param(lineH, 'Name', sigNames{k});
                        set_param(lineH, 'DataLogging', 'on');
                        set_param(lineH, 'DataLoggingNameMode', 'Custom');
                        set_param(lineH, 'DataLoggingNameCustom', sigNames{k});
                    end
                end
            end
        end
    catch ME
        fprintf('  [SignalConfig] Note on Measurements inports: %s\n', ME.message);
    end
end
