function export_simscape_fluids_model(targetPath)
% EXPORT_SIMSCAPE_FLUIDS_MODEL
% Adapts and packages the official MathWorks Simscape Fluids shipping example
% ('ssc_fluids_data_center_cooling') for automated scenario testing and MathWorks
% Excellence in Innovation submission.
%
% Usage:
%   export_simscape_fluids_model();
%   export_simscape_fluids_model('models/DataCenterCooling.slx');

    if nargin < 1 || isempty(targetPath)
        targetPath = fullfile('models', 'DataCenterCooling.slx');
    end

    % 1. Check for Simscape Fluids toolbox availability
    hasFluids = ~isempty(which('ssc_fluids_data_center_cooling'));

    if ~hasFluids
        warning(['Simscape Fluids example "ssc_fluids_data_center_cooling" was not found on MATLAB path.\n' ...
                 'To use the full physical Simscape Fluids model, install Simscape Fluids via:\n' ...
                 '  MATLAB Home Tab -> Add-Ons -> Get Add-Ons -> Search "Simscape Fluids".\n\n' ...
                 'Falling back to building the clean standard Simulink physical plant via build_datacenter_simulink_model()...']);
        build_datacenter_simulink_model(targetPath);
        return;
    end

    fprintf('Loading MathWorks Simscape Fluids Data Center Cooling example...\n');
    openExample('hydro/DataCenterCoolingExample');
    
    sourceModel = 'ssc_fluids_data_center_cooling';
    if ~bdIsLoaded(sourceModel)
        load_system(sourceModel);
    end

    % 2. Ensure target directory exists
    [targetDir, modelName, ~] = fileparts(targetPath);
    if ~isempty(targetDir) && ~exist(targetDir, 'dir')
        mkdir(targetDir);
    end

    % 3. Configure logging and solver settings
    set_param(sourceModel, ...
        'SignalLogging', 'on', ...
        'SignalLoggingName', 'logsout', ...
        'ReturnWorkspaceOutputs', 'on');

    % Save copy as target model name
    save_system(sourceModel, targetPath);
    fprintf('Successfully configured and saved Simscape Fluids model to: %s\n', targetPath);
end
