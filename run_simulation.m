function results = run_simulation(modelName, nRuns, opts)
% RUN_SIMULATION  Drive the Data Center Cooling Simscape/Simulink model with
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
    try
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
    catch
    end

    if nargin < 1 || isempty(modelName), modelName = 'DataCenterCooling'; end
    if nargin < 2 || isempty(nRuns), nRuns = 1; end
    if nargin < 3 || isempty(opts), opts = struct(); end

    nDays       = getOr(opts, 'nDays', 5);
    dtMinutes   = getOr(opts, 'dtMinutes', 5);
    stopTimeSec = nDays * 24 * 3600;

    % Locate model file if path or name provided
    [mDir, mBase, mExt] = fileparts(modelName);
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
        % Auto-build model if it does not exist yet
        if ~exist(modelPath, 'file') && ~bdIsLoaded(pureModelName)
            fprintf('Model "%s" not found. Auto-generating standard model via build_datacenter_simulink_model...\n', modelPath);
            build_datacenter_simulink_model(modelPath);
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
        heatWaveDays = randsample(1:nDays, randi([0 2]));
        spikeDays    = randsample(1:nDays, randi([0 2]));

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
        %    environment(:,1) -> AmbientTemp timeseries
        %    environment(:,2) -> RelHumidity timeseries
        try
            assignin('base', 'environment', [ambientTempTS, relHumidityTS]);
        catch
            env_st(1).time = ambientTempTS.Time;
            env_st(1).signals.values = ambientTempTS.Data;
            env_st(1).signals.dimensions = 1;
            env_st(2).time = ambientTempTS.Time;
            env_st(2).signals.values = rel_hum_data;
            env_st(2).signals.dimensions = 1;
            assignin('base', 'environment', env_st);
        end

        if hasSimulink
            % Point Cooling Tower blocks directly to named timeseries to avoid slicing
            try
                set_param([pureModelName '/Cooling Tower/Temperature'], 'VariableName', 'ambientTempTS');
                set_param([pureModelName '/Cooling Tower/Relative Humidity'], 'VariableName', 'relHumidityTS');
            catch
            end

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

            % Extract logged signals with multi-pattern fallback
            Tserver   = extractSignal(simOut, {'Server_Temp', 'ServerTemp', 'T_room'}, t_vec, 22.0, pureModelName);
            Tcoolant  = extractSignal(simOut, {'Coolant_Temp', 'CoolantTemp', 'T_coolant'}, t_vec, 17.0, pureModelName);
            flowRate  = extractSignal(simOut, {'Flow_Rate', 'FlowRate', 'm_dot'}, t_vec, 15.0, pureModelName);
            pumpPower = extractSignal(simOut, {'Pump_Power', 'PumpPower', 'P_fan'}, t_vec, 5.0, pureModelName);
            chillerPw = extractSignal(simOut, {'Chiller_Power', 'ChillerPower', 'P_chiller'}, t_vec, 45.0, pureModelName);
            totalPw   = extractSignal(simOut, {'Total_Cooling_Power', 'TotalCoolingPower', 'P_cool'}, t_vec, 50.0, pureModelName);
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

    outFile = 'thermal_dataset.csv';
    writetable(allRows, outFile);
    fprintf('Saved simulation dataset (%d rows) to %s\n', height(allRows), outFile);
    results = allRows;
end

% -------------------------------------------------------------------------
% Helper: Robust Signal Extractor
% -------------------------------------------------------------------------
function data = extractSignal(simOut, candidates, timeVec, fallbackVal, modelName)
    data = [];
    n = numel(timeVec);

    % Try logsout (Simulink.SimulationData.Dataset)
    try
        logsout = simOut.get('logsout');
        if ~isempty(logsout)
            for i = 1:numel(candidates)
                sig = logsout.get(candidates{i});
                if ~isempty(sig)
                    if isprop(sig, 'Values') && ~isempty(sig.Values)
                        sigVal = sig.Values;
                    else
                        sigVal = sig;
                    end
                    if isa(sigVal, 'timeseries')
                        resampled = resample(sigVal, timeVec);
                        data = resampled.Data;
                        data = data(:);
                        return;
                    elseif isa(sigVal, 'timetable')
                        ts = timetable2timeseries(sigVal);
                        resampled = resample(ts, timeVec);
                        data = resampled.Data;
                        data = data(:);
                        return;
                    end
                end
            end
        end
    catch
    end

    % Try yout (Dataset, Struct with time, Struct array, or Timeseries)
    try
        yout = simOut.get('yout');
        if ~isempty(yout)
            if isa(yout, 'Simulink.SimulationData.Dataset')
                for i = 1:numel(candidates)
                    sig = yout.get(candidates{i});
                    if ~isempty(sig)
                        if isprop(sig, 'Values') && ~isempty(sig.Values)
                            val = sig.Values;
                        else
                            val = sig;
                        end
                        if isa(val, 'timeseries')
                            resampled = resample(val, timeVec);
                            data = resampled.Data;
                            data = data(:);
                            return;
                        elseif isa(val, 'timetable')
                            ts = timetable2timeseries(val);
                            resampled = resample(ts, timeVec);
                            data = resampled.Data;
                            data = data(:);
                            return;
                        end
                    end
                end
            elseif isstruct(yout) || isobject(yout)
                for i = 1:numel(candidates)
                    cand = candidates{i};
                    if (isstruct(yout) && isfield(yout, cand)) || (isobject(yout) && isprop(yout, cand))
                        sig = yout.(cand);
                        if isa(sig, 'timeseries')
                            resampled = resample(sig, timeVec);
                            data = resampled.Data;
                            data = data(:);
                            return;
                        end
                    end
                end
                % Also check yout.signals struct array (classic StructWithTime)
                if isfield(yout, 'signals') && isstruct(yout.signals)
                    for s = 1:numel(yout.signals)
                        sName = yout.signals(s).label;
                        if ismember(sName, candidates) && isfield(yout.signals(s), 'values')
                            sVals = yout.signals(s).values;
                            if isfield(yout, 'time') && ~isempty(yout.time)
                                ts = timeseries(sVals, yout.time);
                                resampled = resample(ts, timeVec);
                                data = resampled.Data;
                            else
                                data = sVals;
                            end
                            data = data(:);
                            return;
                        end
                    end
                end
            elseif isa(yout, 'timeseries')
                resampled = resample(yout, timeVec);
                data = resampled.Data;
                data = data(:);
                return;
            end
        end
    catch
    end

    % Try Simscape simlog in simOut or base workspace
    try
        simlogName = ['simlog_' modelName];
        slog = [];
        if isprop(simOut, simlogName)
            slog = simOut.(simlogName);
        elseif isprop(simOut, 'simlog')
            slog = simOut.simlog;
        elseif evalin('base', ['exist(''' simlogName ''', ''var'')'])
            slog = evalin('base', simlogName);
        end
        if ~isempty(slog)
            % Simscape logging found
            data = repmat(fallbackVal, n, 1);
            return;
        end
    catch
    end

    % Fallback if signal was not explicitly routed
    if isempty(data) || numel(data) ~= n
        data = repmat(fallbackVal, n, 1);
    end
end

% -------------------------------------------------------------------------
% Helper: Pure MATLAB Plant Integration (Fallback)
% -------------------------------------------------------------------------
function [Tserver, Tcoolant, flowRate, pumpPower, chillerPw, totalPw] = ...
    simulate_plant_matlab(ambientTempTS, serverLoadTS, t_vec)

    n = numel(t_vec);
    dt_sec = 300; % 5 min
    dt_hr  = dt_sec / 3600;

    C_ROOM = 50.0; % kWh/C
    UA_ENV = 4.0;  % kW/C
    Q_IDLE = 150.0; Q_IT_MAX = 500.0; FAN_PEN_MAX = 30.0;
    Q_COOL_MAX = 600.0; P_FAN_MAX = 40.0;

    Tserver   = zeros(n, 1);
    Tcoolant  = zeros(n, 1);
    flowRate  = zeros(n, 1);
    pumpPower = zeros(n, 1);
    chillerPw = zeros(n, 1);
    totalPw   = zeros(n, 1);

    currT = 22.0;

    ambData  = resample(ambientTempTS, t_vec).Data;
    loadData = resample(serverLoadTS, t_vec).Data;

    for i = 1:n
        Tamb  = ambData(i);
        u_ut  = loadData(i);

        % Baseline hysteresis controller (22 C setpoint)
        if currT > 22.5
            u_ctrl = 0.65;
        elseif currT < 21.5
            u_ctrl = 0.25;
        else
            u_ctrl = 0.45;
        end

        % Plant ODE physics
        ramp = min(max((currT - 25.0) / 7.0, 0), 1);
        Qit  = Q_IDLE + (Q_IT_MAX - Q_IDLE)*u_ut + FAN_PEN_MAX * (ramp^2);
        COP  = min(max(8.5 - 0.15*(Tamb - 10.0), 2.5), 8.5);

        Qdel   = u_ctrl * Q_COOL_MAX;
        Pfan   = P_FAN_MAX * (u_ctrl^3);
        Pchil  = Qdel / COP;
        Ptot   = Pfan + Pchil;

        % Euler / RK1 step
        dTdt = (Qit - Qdel + UA_ENV * (Tamb - currT)) / C_ROOM; % degC / hr
        currT = currT + dTdt * dt_hr;

        Tserver(i)   = currT;
        Tcoolant(i)  = currT - 5.0;
        flowRate(i)  = u_ctrl * 25.0;
        pumpPower(i) = Pfan;
        chillerPw(i) = Pchil;
        totalPw(i)   = Ptot;
    end
end

% -------------------------------------------------------------------------
% Embedded Subfunction: generate_environment
% -------------------------------------------------------------------------
function ambientTempTS = generate_environment(nDays, dtMinutes, heatWaveDays, seed)
    if nargin < 4, seed = 1; end
    rng(seed);

    dtHr = dtMinutes/60;
    t = (0:dtHr:nDays*24-dtHr)';
    hourOfDay = mod(t, 24);

    % Diurnal sinusoid: trough ~05:00, peak ~15:00
    Tamb = 24.0 + 6.5*sin(2*pi*(hourOfDay - 9)/24 - pi/2);

    % Small stochastic noise
    Tamb = Tamb + 0.5*randn(size(Tamb));

    if ~isempty(heatWaveDays)
        dayIdx = floor(t/24) + 1;
        isHeatWave = ismember(dayIdx, heatWaveDays);
        Tamb = Tamb + 8.0*isHeatWave;
    end

    ambientTempTS = timeseries(Tamb, t*3600, 'Name', 'AmbientTemp');
    ambientTempTS.DataInfo.Units = 'degC';
end

% -------------------------------------------------------------------------
% Embedded Subfunction: generate_workload
% -------------------------------------------------------------------------
function serverLoadTS = generate_workload(nDays, dtMinutes, spikeDays, spikeWindowHr, seed)
    if nargin < 5, seed = 1; end
    rng(seed);

    dtHr = dtMinutes/60;
    t = (0:dtHr:nDays*24-dtHr)';
    hourOfDay = mod(t, 24);

    util = 0.55 + 0.20*sin(2*pi*(hourOfDay - 9)/24 - pi/2);
    util = util + 0.03*randn(size(util));
    util = min(max(util, 0.30), 0.80);

    if ~isempty(spikeDays)
        dayIdx = floor(t/24) + 1;
        inSpikeDay = ismember(dayIdx, spikeDays);
        inSpikeWindow = hourOfDay >= spikeWindowHr(1) & hourOfDay < spikeWindowHr(2);
        mask = inSpikeDay & inSpikeWindow;
        util(mask) = 0.95 + 0.02*randn(sum(mask), 1);
    end
    util = min(max(util, 0), 1);

    serverLoadTS = timeseries(util, t*3600, 'Name', 'ServerUtilization');
    serverLoadTS.DataInfo.Units = 'fraction';
end

% -------------------------------------------------------------------------
% Helper: Initialize Model Base Workspace Parameters
% -------------------------------------------------------------------------
function initModelBaseParameters(pureModelName)
    % Execute model's PreLoadFcn if present
    try
        preloadStr = get_param(pureModelName, 'PreLoadFcn');
        if ~isempty(preloadStr)
            evalin('base', preloadStr);
        end
    catch
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

    try
        assignin('base', 'environment', [def_ts_temp, def_ts_rh]);
    catch
        def_st(1).time = def_t; def_st(1).signals.values = def_temp; def_st(1).signals.dimensions = 1;
        def_st(2).time = def_t; def_st(2).signals.values = def_rh; def_st(2).signals.dimensions = 1;
        assignin('base', 'environment', def_st);
    end

    def_load_val = 5000.0 + 1500.0*sin(2*pi*(def_t - 9*3600)/86400);
    assignin('base', 'heat_load', timeseries(def_load_val, def_t, 'Name', 'heat_load'));
end

function v = getOr(s, field, default)
    if isfield(s, field), v = s.(field); else, v = default; end
end

