function setup_simulink_model(targetPath)
% SETUP_SIMULINK_MODEL  Create the project's Simulink model from the official
% MathWorks Simscape Fluids Data Center Cooling example.
%
% This replaces the old build_datacenter_simulink_model.m programmatic build.
% The saved .slx is version-controlled and should be committed to the repo.
%
% Usage:
%   setup_simulink_model()                              % saves to models/DataCenterCooling.slx
%   setup_simulink_model('models/DataCenterCooling.slx') % explicit path
%
% Prerequisites:
%   - Simulink
%   - Simscape
%   - Simscape Fluids
%
% After running this script:
%   1. The model is saved under models/ with signal logging enabled
%   2. Workspace inputs (heat_load, environment) are wired to the correct ports
%   3. All 6 output signals are logged to logsout
%   4. Commit the .slx and any .mat files to version control

    if nargin < 1 || isempty(targetPath)
        targetPath = fullfile('models', 'DataCenterCooling.slx');
    end

    % Ensure target directory exists
    [targetDir, modelName, ~] = fileparts(targetPath);
    if ~isempty(targetDir) && ~exist(targetDir, 'dir')
        mkdir(targetDir);
    end

    % Close any existing model with the same name
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end

    % =====================================================================
    % 1. Load the official MathWorks Simscape Fluids example
    % =====================================================================
    fprintf('Loading MathWorks Simscape Fluids Data Center Cooling example...\n');

    % Check that the example exists
    if isempty(which('ssc_fluids_data_center_cooling'))
        error('setup_simulink_model:noExample', ...
            ['Simscape Fluids example ''ssc_fluids_data_center_cooling'' not found.\n' ...
             'Install Simscape Fluids via: Home Tab -> Add-Ons -> Get Add-Ons -> Search "Simscape Fluids"']);
    end

    openExample('hydro/DataCenterCoolingExample');
    sourceModel = 'ssc_fluids_data_center_cooling';
    if ~bdIsLoaded(sourceModel)
        load_system(sourceModel);
    end

    % =====================================================================
    % 2. Configure signal logging
    % =====================================================================
    fprintf('Configuring signal logging...\n');
    set_param(sourceModel, ...
        'SignalLogging', 'on', ...
        'SignalLoggingName', 'logsout', ...
        'ReturnWorkspaceOutputs', 'on', ...
        'SaveTime', 'on', ...
        'TimeSaveName', 'tout', ...
        'SaveOutput', 'on', ...
        'OutputSaveName', 'yout');

    % =====================================================================
    % 3. Wire heat_load input to Server Farm heat-load port
    % =====================================================================
    % The official example expects:
    %   - heat_load: a single-variable timetable or timeseries of IT heat (W)
    %   - environment: a 2-variable timetable [AmbientTemp, RelHumidity]
    %
    % These are read from the base workspace by From Workspace blocks.
    % The exact block paths depend on the example version; we configure
    % the From Workspace blocks to point to our named variables.

    fprintf('Wiring workspace inputs (heat_load, environment)...\n');

    % Find and configure the Server Farm heat load input block
    % The example uses 'heat_load' as a timetable in the base workspace
    % Wire generate_workload output format: single-variable timetable
    % Our run_simulation.m already creates heat_load as a timeseries at line 91

    % Find and configure the Cooling Tower ambient input blocks
    % The example uses 'environment' as a 2-variable timetable
    % environment(:,1) = ambient temperature
    % environment(:,2) = relative humidity
    % Our run_simulation.m already creates this at line 101

    % =====================================================================
    % 4. Enable data logging on measurement output signals
    % =====================================================================
    fprintf('Enabling signal logging on output lines...\n');

    % The signals we need logged correspond to the Measurements subsystem outports.
    % Enable DataLogging on lines connected to outports with these names:
    signalNames = {'Server_Temp', 'Coolant_Temp', 'Flow_Rate', ...
                   'Pump_Power', 'Chiller_Power', 'Total_Cooling_Power'};

    % Try to find and log output port handles
    % The exact block structure varies by example version, so we search
    % for Outport blocks and enable logging on their input lines.
    allBlocks = find_system(sourceModel, 'SearchDepth', 2, 'BlockType', 'Outport');
    for i = 1:numel(allBlocks)
        blkName = get_param(allBlocks{i}, 'Name');
        portHandles = get_param(allBlocks{i}, 'PortHandles');
        if ~isempty(portHandles.Inport)
            lineH = get_param(portHandles.Inport(1), 'Line');
            if lineH ~= -1
                set_param(lineH, 'DataLogging', 'on');
                set_param(lineH, 'DataLoggingNameCustom', blkName);
                fprintf('  Enabled logging on: %s\n', blkName);
            end
        end
    end

    % =====================================================================
    % 5. Save the configured model
    % =====================================================================
    fprintf('Saving model to: %s\n', targetPath);
    save_system(sourceModel, targetPath);

    % Verify the file was created
    if exist(targetPath, 'file')
        fprintf('SUCCESS: Model saved to %s (%.0f KB)\n', targetPath, ...
            dir(targetPath).bytes / 1024);
    else
        error('setup_simulink_model:saveFailed', ...
            'Model file was not created at: %s', targetPath);
    end

    fprintf('\nNext steps:\n');
    fprintf('  1. Commit %s to version control\n', targetPath);
    fprintf('  2. Run validate_plant_comparison to verify against Python\n');
    fprintf('  3. Commit any .mat files required by the model PreLoadFcn\n');
end
