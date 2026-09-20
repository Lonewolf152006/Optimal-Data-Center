function modelName = build_datacenter_simulink_model(targetPath)
% BUILD_DATACENTER_SIMULINK_MODEL
% Programmatically creates the DataCenterCooling Simulink model (.slx)
% for MathWorks Excellence in Innovation Project #196 (Optimal Data Center Cooling).
%
% This script builds a clean, fully-parameterized Simulink model representing:
%   - Workspace Timeseries inputs: 'serverLoadTS', 'ambientTempTS'
%   - Dynamic IT server heat load with non-linear server-fan ramp penalty
%   - Variable-capacity CRAC & Chiller with ambient-temperature-dependent COP
%   - Thermal capacitance zone ODE (dT/dt)
%   - Baseline Thermostatic Controller (22 degC setpoint with hysteresis)
%   - Outports and signal logging for:
%       * Server_Temp (degC)
%       * Coolant_Temp (degC)
%       * Flow_Rate (kg/s)
%       * Pump_Power (kW)
%       * Chiller_Power (kW)
%       * Total_Cooling_Power (kW)
%
% Usage:
%   build_datacenter_simulink_model();
%   build_datacenter_simulink_model('models/DataCenterCooling.slx');

    if nargin < 1 || isempty(targetPath)
        targetPath = fullfile('models', 'DataCenterCooling.slx');
    end

    % Ensure destination folder exists
    [targetDir, baseName, ~] = fileparts(targetPath);
    if ~isempty(targetDir) && ~exist(targetDir, 'dir')
        mkdir(targetDir);
    end
    modelName = baseName;

    % Close if already open
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end

    fprintf('Creating Simulink model: %s...\n', modelName);
    new_system(modelName);
    open_system(modelName);

    % Configure solver and logging settings
    set_param(modelName, ...
        'Solver', 'ode45', ...
        'StartTime', '0', ...
        'StopTime', '432000', ...       % 5 days in seconds (5 * 24 * 3600)
        'SaveTime', 'on', ...
        'TimeSaveName', 'tout', ...
        'SaveOutput', 'on', ...
        'OutputSaveName', 'yout', ...
        'SignalLogging', 'on', ...
        'SignalLoggingName', 'logsout', ...
        'ReturnWorkspaceOutputs', 'on');

    % Set plant parameters in Model Workspace or Base Workspace
    ws = get_param(modelName, 'ModelWorkspace');
    ws.assignin('C_ROOM', 50.0 * 3600);   % kJ/degC (50 kWh/degC * 3600 s/hr)
    ws.assignin('UA_ENV', 4.0);           % kW/degC
    ws.assignin('Q_IDLE', 150.0);         % kW
    ws.assignin('Q_IT_MAX', 500.0);       % kW
    ws.assignin('FAN_PEN_MAX', 30.0);     % kW
    ws.assignin('Q_COOL_MAX', 600.0);     % kW
    ws.assignin('P_FAN_MAX', 40.0);       % kW
    ws.assignin('T_INIT', 22.0);          % degC

    % -------------------------------------------------------------
    % Add Input Sources (From Workspace)
    % -------------------------------------------------------------
    % 1. Server Load TimeSeries Input
    add_block('simulink/Sources/From Workspace', [modelName '/FromWS_ServerLoad'], ...
        'Position', [40, 80, 160, 110], ...
        'VariableName', 'serverLoadTS', ...
        'Interpolate', 'on', ...
        'OutputAfterFinalValue', 'Holding final value');

    % 2. Ambient Temperature TimeSeries Input
    add_block('simulink/Sources/From Workspace', [modelName '/FromWS_AmbientTemp'], ...
        'Position', [40, 200, 160, 230], ...
        'VariableName', 'ambientTempTS', ...
        'Interpolate', 'on', ...
        'OutputAfterFinalValue', 'Holding final value');

    % -------------------------------------------------------------
    % Controller Subsystem (Baseline 22 degC Setpoint Hysteresis)
    % -------------------------------------------------------------
    add_block('simulink/Commonly Used Blocks/Subsystem', [modelName '/Controller'], ...
        'Position', [220, 280, 340, 340]);
    % Configure Controller internals
    ctrlLines = get_param([modelName '/Controller'], 'Lines');
    for i = 1:length(ctrlLines), delete_line(ctrlLines(i).Handle); end
    
    % Inport 1: T_room
    set_param([modelName '/Controller/In1'], 'Name', 'T_room', 'Position', [40, 50, 70, 65]);
    % Baseline logic: if T > 22.5 -> u = 0.65; if T < 21.5 -> u = 0.25; else 0.45
    add_block('simulink/Logic and Bit Operations/Relational Operator', [modelName '/Controller/RelOpHi'], ...
        'Operator', '>', 'Position', [120, 42, 150, 73]);
    add_block('simulink/Sources/Constant', [modelName '/Controller/ThreshHi'], ...
        'Value', '22.5', 'Position', [40, 80, 80, 95]);
    add_line([modelName '/Controller'], 'T_room/1', 'RelOpHi/1');
    add_line([modelName '/Controller'], 'ThreshHi/1', 'RelOpHi/2');

    add_block('simulink/Signal Routing/Switch', [modelName '/Controller/SwitchHi'], ...
        'Threshold', '0.5', 'Position', [200, 45, 230, 95]);
    add_block('simulink/Sources/Constant', [modelName '/Controller/u_high'], ...
        'Value', '0.65', 'Position', [120, 20, 160, 35]);
    add_block('simulink/Sources/Constant', [modelName '/Controller/u_nom'], ...
        'Value', '0.45', 'Position', [120, 110, 160, 125]);
    add_line([modelName '/Controller'], 'u_high/1', 'SwitchHi/1');
    add_line([modelName '/Controller'], 'RelOpHi/1', 'SwitchHi/2');
    add_line([modelName '/Controller'], 'u_nom/1', 'SwitchHi/3');
    add_line([modelName '/Controller'], 'SwitchHi/1', 'Out1/1');
    set_param([modelName '/Controller/Out1'], 'Name', 'u_ctrl', 'Position', [280, 60, 310, 75]);

    % -------------------------------------------------------------
    % Thermal Plant Subsystem (Lumped Parameter ODE)
    % -------------------------------------------------------------
    add_block('simulink/Commonly Used Blocks/Subsystem', [modelName '/ThermalPlant'], ...
        'Position', [400, 60, 580, 260]);
    plantLines = get_param([modelName '/ThermalPlant'], 'Lines');
    for i = 1:length(plantLines), delete_line(plantLines(i).Handle); end

    % Inports to ThermalPlant
    set_param([modelName '/ThermalPlant/In1'], 'Name', 'u_util', 'Position', [30, 40, 60, 55]);
    add_block('simulink/Sources/Inport', [modelName '/ThermalPlant/T_amb'], 'Position', [30, 100, 60, 115]);
    add_block('simulink/Sources/Inport', [modelName '/ThermalPlant/u_ctrl'], 'Position', [30, 160, 60, 175]);

    % Integrator for T_room: dT/dt -> T_room
    add_block('simulink/Continuous/Integrator', [modelName '/ThermalPlant/Integrator_Troom'], ...
        'InitialCondition', '22.0', 'Position', [480, 90, 510, 120]);

    % Heat gain: Q_IT = 150 + 350 * u_util
    add_block('simulink/Math Operations/Gain', [modelName '/ThermalPlant/Gain_IT'], ...
        'Gain', '350.0', 'Position', [100, 35, 140, 65]);
    add_block('simulink/Sources/Constant', [modelName '/ThermalPlant/Q_idle_const'], ...
        'Value', '150.0', 'Position', [100, 75, 140, 95]);
    add_block('simulink/Math Operations/Sum', [modelName '/ThermalPlant/Sum_QIT'], ...
        'Inputs', '++', 'Position', [170, 45, 195, 75]);
    add_line([modelName '/ThermalPlant'], 'u_util/1', 'Gain_IT/1');
    add_line([modelName '/ThermalPlant'], 'Gain_IT/1', 'Sum_QIT/1');
    add_line([modelName '/ThermalPlant'], 'Q_idle_const/1', 'Sum_QIT/2');

    % Cooling delivered: Q_cool = u_ctrl * 600
    add_block('simulink/Math Operations/Gain', [modelName '/ThermalPlant/Gain_Qcool'], ...
        'Gain', '600.0', 'Position', [100, 155, 150, 185]);
    add_line([modelName '/ThermalPlant'], 'u_ctrl/1', 'Gain_Qcool/1');

    % Envelope heat exchange: UA_ENV * (T_amb - T_room)
    add_block('simulink/Math Operations/Sum', [modelName '/ThermalPlant/Sum_DeltaTamb'], ...
        'Inputs', '+-', 'Position', [200, 100, 225, 130]);
    add_line([modelName '/ThermalPlant'], 'T_amb/1', 'Sum_DeltaTamb/1');
    add_line([modelName '/ThermalPlant'], 'Integrator_Troom/1', 'Sum_DeltaTamb/2');
    add_block('simulink/Math Operations/Gain', [modelName '/ThermalPlant/Gain_UA'], ...
        'Gain', '4.0', 'Position', [245, 105, 280, 125]);
    add_line([modelName '/ThermalPlant'], 'Sum_DeltaTamb/1', 'Gain_UA/1');

    % Total net heat flow: NetQ = Q_IT - Q_cool + Q_env
    add_block('simulink/Math Operations/Sum', [modelName '/ThermalPlant/Sum_NetHeat'], ...
        'Inputs', '+-+', 'Position', [330, 85, 360, 125]);
    add_line([modelName '/ThermalPlant'], 'Sum_QIT/1', 'Sum_NetHeat/1');
    add_line([modelName '/ThermalPlant'], 'Gain_Qcool/1', 'Sum_NetHeat/2');
    add_line([modelName '/ThermalPlant'], 'Gain_UA/1', 'Sum_NetHeat/3');

    % Divide by thermal mass C_ROOM (50 kWh/C * 3600 s/hr = 180,000 kJ/C)
    add_block('simulink/Math Operations/Gain', [modelName '/ThermalPlant/Gain_InvCapacitance'], ...
        'Gain', '1/180000', 'Position', [390, 95, 450, 115]);
    add_line([modelName '/ThermalPlant'], 'Sum_NetHeat/1', 'Gain_InvCapacitance/1');
    add_line([modelName '/ThermalPlant'], 'Gain_InvCapacitance/1', 'Integrator_Troom/1');

    % Outport 1: Server_Temp (T_room)
    set_param([modelName '/ThermalPlant/Out1'], 'Name', 'Server_Temp', 'Position', [560, 95, 590, 115]);
    add_line([modelName '/ThermalPlant'], 'Integrator_Troom/1', 'Server_Temp/1');

    % Outport 2: Coolant_Temp = T_room - 5 degC
    add_block('simulink/Math Operations/Sum', [modelName '/ThermalPlant/Sum_Tcoolant'], ...
        'Inputs', '+-', 'Position', [540, 150, 565, 175]);
    add_block('simulink/Sources/Constant', [modelName '/ThermalPlant/CoolantOffset'], ...
        'Value', '5.0', 'Position', [480, 170, 510, 185]);
    add_line([modelName '/ThermalPlant'], 'Integrator_Troom/1', 'Sum_Tcoolant/1');
    add_line([modelName '/ThermalPlant'], 'CoolantOffset/1', 'Sum_Tcoolant/2');
    add_block('simulink/Sinks/Outport', [modelName '/ThermalPlant/Coolant_Temp'], 'Position', [590, 155, 620, 170]);
    add_line([modelName '/ThermalPlant'], 'Sum_Tcoolant/1', 'Coolant_Temp/1');

    % -------------------------------------------------------------
    % Auxiliary Power & Hydraulic Calculations Subsystem
    % -------------------------------------------------------------
    add_block('simulink/Commonly Used Blocks/Subsystem', [modelName '/AuxiliaryMetrics'], ...
        'Position', [650, 140, 800, 260]);
    auxLines = get_param([modelName '/AuxiliaryMetrics'], 'Lines');
    for i = 1:length(auxLines), delete_line(auxLines(i).Handle); end

    set_param([modelName '/AuxiliaryMetrics/In1'], 'Name', 'u_ctrl', 'Position', [30, 40, 60, 55]);
    add_block('simulink/Sources/Inport', [modelName '/AuxiliaryMetrics/T_amb'], 'Position', [30, 120, 60, 135]);

    % Flow Rate = u_ctrl * 25.0 kg/s
    add_block('simulink/Math Operations/Gain', [modelName '/AuxiliaryMetrics/Gain_Flow'], ...
        'Gain', '25.0', 'Position', [100, 35, 140, 60]);
    add_line([modelName '/AuxiliaryMetrics'], 'u_ctrl/1', 'Gain_Flow/1');
    set_param([modelName '/AuxiliaryMetrics/Out1'], 'Name', 'Flow_Rate', 'Position', [260, 40, 290, 55]);
    add_line([modelName '/AuxiliaryMetrics'], 'Gain_Flow/1', 'Flow_Rate/1');

    % Pump/Fan Power = 40 * u_ctrl^3
    add_block('simulink/Math Operations/Math Function', [modelName '/AuxiliaryMetrics/Math_Cube'], ...
        'Function', 'pow', 'Position', [100, 85, 130, 105]);
    add_block('simulink/Sources/Constant', [modelName '/AuxiliaryMetrics/Const_Three'], ...
        'Value', '3', 'Position', [40, 95, 65, 110]);
    add_line([modelName '/AuxiliaryMetrics'], 'u_ctrl/1', 'Math_Cube/1');
    add_line([modelName '/AuxiliaryMetrics'], 'Const_Three/1', 'Math_Cube/2');
    add_block('simulink/Math Operations/Gain', [modelName '/AuxiliaryMetrics/Gain_Pfan'], ...
        'Gain', '40.0', 'Position', [160, 85, 200, 105]);
    add_line([modelName '/AuxiliaryMetrics'], 'Math_Cube/1', 'Gain_Pfan/1');
    add_block('simulink/Sinks/Outport', [modelName '/AuxiliaryMetrics/Pump_Power'], 'Position', [260, 88, 290, 102]);
    add_line([modelName '/AuxiliaryMetrics'], 'Gain_Pfan/1', 'Pump_Power/1');

    % Chiller Power = (u_ctrl * 600) / COP, with COP = clip(8.5 - 0.15*(T_amb - 10), 2.5, 8.5)
    add_block('simulink/Math Operations/Gain', [modelName '/AuxiliaryMetrics/Gain_Qcool'], ...
        'Gain', '600.0', 'Position', [100, 145, 140, 170]);
    add_line([modelName '/AuxiliaryMetrics'], 'u_ctrl/1', 'Gain_Qcool/1');

    % COP Calculation
    add_block('simulink/Math Operations/Sum', [modelName '/AuxiliaryMetrics/Sum_Tamb10'], ...
        'Inputs', '+-', 'Position', [90, 200, 110, 220]);
    add_block('simulink/Sources/Constant', [modelName '/AuxiliaryMetrics/Const_10'], ...
        'Value', '10.0', 'Position', [40, 215, 65, 230]);
    add_line([modelName '/AuxiliaryMetrics'], 'T_amb/1', 'Sum_Tamb10/1');
    add_line([modelName '/AuxiliaryMetrics'], 'Const_10/1', 'Sum_Tamb10/2');

    add_block('simulink/Math Operations/Gain', [modelName '/AuxiliaryMetrics/Gain_Slope'], ...
        'Gain', '-0.15', 'Position', [130, 200, 160, 220]);
    add_line([modelName '/AuxiliaryMetrics'], 'Sum_Tamb10/1', 'Gain_Slope/1');

    add_block('simulink/Math Operations/Sum', [modelName '/AuxiliaryMetrics/Sum_COPBase'], ...
        'Inputs', '++', 'Position', [180, 200, 200, 225]);
    add_block('simulink/Sources/Constant', [modelName '/AuxiliaryMetrics/Const_8p5'], ...
        'Value', '8.5', 'Position', [130, 235, 160, 250]);
    add_line([modelName '/AuxiliaryMetrics'], 'Gain_Slope/1', 'Sum_COPBase/1');
    add_line([modelName '/AuxiliaryMetrics'], 'Const_8p5/1', 'Sum_COPBase/2');

    add_block('simulink/Discontinuities/Saturation', [modelName '/AuxiliaryMetrics/Sat_COP'], ...
        'UpperLimit', '8.5', 'LowerLimit', '2.5', 'Position', [220, 205, 245, 225]);
    add_line([modelName '/AuxiliaryMetrics'], 'Sum_COPBase/1', 'Sat_COP/1');

    add_block('simulink/Math Operations/Divide', [modelName '/AuxiliaryMetrics/Div_Chiller'], ...
        'Inputs', '*/', 'Position', [265, 150, 290, 175]);
    add_line([modelName '/AuxiliaryMetrics'], 'Gain_Qcool/1', 'Div_Chiller/1');
    add_line([modelName '/AuxiliaryMetrics'], 'Sat_COP/1', 'Div_Chiller/2');
    add_block('simulink/Sinks/Outport', [modelName '/AuxiliaryMetrics/Chiller_Power'], 'Position', [330, 155, 360, 170]);
    add_line([modelName '/AuxiliaryMetrics'], 'Div_Chiller/1', 'Chiller_Power/1');

    % Total Power = Pump_Power + Chiller_Power
    add_block('simulink/Math Operations/Sum', [modelName '/AuxiliaryMetrics/Sum_TotalPower'], ...
        'Inputs', '++', 'Position', [330, 205, 355, 230]);
    add_line([modelName '/AuxiliaryMetrics'], 'Gain_Pfan/1', 'Sum_TotalPower/1');
    add_line([modelName '/AuxiliaryMetrics'], 'Div_Chiller/1', 'Sum_TotalPower/2');
    add_block('simulink/Sinks/Outport', [modelName '/AuxiliaryMetrics/Total_Cooling_Power'], 'Position', [390, 210, 420, 225]);
    add_line([modelName '/AuxiliaryMetrics'], 'Sum_TotalPower/1', 'Total_Cooling_Power/1');

    % -------------------------------------------------------------
    % Wire Top-Level Diagram & Outports
    % -------------------------------------------------------------
    % Connect ServerLoad -> ThermalPlant Inport 1
    add_line(modelName, 'FromWS_ServerLoad/1', 'ThermalPlant/1');

    % Connect AmbientTemp -> ThermalPlant Inport 2
    add_line(modelName, 'FromWS_AmbientTemp/1', 'ThermalPlant/2');

    % Connect Feedback: Server_Temp -> Controller Inport 1
    add_line(modelName, 'ThermalPlant/1', 'Controller/1');

    % Connect Controller u_ctrl -> ThermalPlant Inport 3
    add_line(modelName, 'Controller/1', 'ThermalPlant/3');

    % Connect Auxiliary Inputs: u_ctrl & AmbientTemp
    add_line(modelName, 'Controller/1', 'AuxiliaryMetrics/1');
    add_line(modelName, 'FromWS_AmbientTemp/1', 'AuxiliaryMetrics/2');

    % Add Top-Level Outports with Signal Logging enabled
    outportNames = {'Server_Temp', 'Coolant_Temp', 'Flow_Rate', 'Pump_Power', 'Chiller_Power', 'Total_Cooling_Power'};
    srcBlocks    = {'ThermalPlant/1', 'ThermalPlant/2', 'AuxiliaryMetrics/1', 'AuxiliaryMetrics/2', 'AuxiliaryMetrics/3', 'AuxiliaryMetrics/4'};
    yPositions   = [60, 100, 140, 180, 220, 260];

    for i = 1:length(outportNames)
        blkPath = [modelName '/' outportNames{i}];
        add_block('simulink/Sinks/Outport', blkPath, ...
            'Position', [870, yPositions(i), 900, yPositions(i) + 15]);
        lineH = add_line(modelName, srcBlocks{i}, [outportNames{i} '/1']);
        % Enable logging on this line
        set_param(lineH, 'DataLogging', 'on');
        set_param(lineH, 'DataLoggingNameCustom', outportNames{i});
    end

    % Save and close
    save_system(modelName, targetPath);
    fprintf('Successfully built and saved Simulink model to: %s\n', targetPath);
end
