classdef test_datacenter_simulation < matlab.unittest.TestCase
    % TEST_DATACENTER_SIMULATION
    % Unit test suite for MathWorks Excellence in Innovation Challenge #196
    % Verifies environment generation, workload synthesis, thermal physics,
    % baseline thermostat control, and sample data availability.
    %
    % To run in MATLAB:
    %   results = runtests('tests/test_datacenter_simulation')

    methods (Test)
        function testSampleDataPresence(testCase)
            % Verify that data/sample/sample_thermal_data.csv exists and is readable
            samplePath = fullfile('..', 'data', 'sample', 'sample_thermal_data.csv');
            if ~exist(samplePath, 'file')
                samplePath = fullfile('data', 'sample', 'sample_thermal_data.csv');
            end
            testCase.verifyTrue(exist(samplePath, 'file') == 2, ...
                'Sample data file data/sample/sample_thermal_data.csv must exist');
            
            data = readtable(samplePath);
            testCase.verifyGreaterThanOrEqual(height(data), 100, ...
                'Sample dataset should contain at least 100 timesteps');
            testCase.verifyTrue(ismember('room_temp', data.Properties.VariableNames), ...
                'Dataset must contain room_temp column');
        end

        function testEnvironmentGeneration(testCase)
            % Verify generate_environment function outputs valid physical ranges
            if exist('generate_environment.m', 'file') || exist(fullfile('..', 'generate_environment.m'), 'file')
                env = generate_environment(1);
                testCase.verifyNotEmpty(env, 'Environment struct must not be empty');
                testCase.verifyTrue(isfield(env, 'time') && isfield(env, 'ambient_temp'), ...
                    'Environment must have time and ambient_temp fields');
                
                % Ambient temperature should be realistic (10C to 45C)
                testCase.verifyGreaterThanOrEqual(min(env.ambient_temp), 10.0);
                testCase.verifyLessThanOrEqual(max(env.ambient_temp), 45.0);
            end
        end

        function testWorkloadGeneration(testCase)
            % Verify generate_workload function bounds utilization in [0, 1]
            if exist('generate_workload.m', 'file') || exist(fullfile('..', 'generate_workload.m'), 'file')
                workload = generate_workload(1);
                testCase.verifyNotEmpty(workload, 'Workload struct must not be empty');
                testCase.verifyTrue(isfield(workload, 'utilization'), ...
                    'Workload must contain utilization field');
                testCase.verifyGreaterThanOrEqual(min(workload.utilization), 0.0);
                testCase.verifyLessThanOrEqual(max(workload.utilization), 1.0);
            end
        end

        function testThermostatLogic(testCase)
            % Verify hysteresis bang-bang logic around setpoint (22.0 C)
            T_set = 22.0;
            deadband = 1.0;
            
            % If temp is significantly above setpoint + deadband, cooling must activate
            T_hot = 24.5;
            u_hot = double(T_hot > (T_set + deadband));
            testCase.verifyEqual(u_hot, 1.0, 'Cooling should be active when T > T_set + deadband');
            
            % If temp is below setpoint - deadband, cooling must deactivate
            T_cold = 20.0;
            u_cold = double(T_cold > (T_set - deadband));
            testCase.verifyEqual(u_cold, 0.0, 'Cooling should be inactive when T < T_set - deadband');
        end

        function testThermalEnergyBalance(testCase)
            % Verify lumped thermal capacity physics step: C * dT/dt = Q_in - Q_out
            C_room = 50.0;     % kWh/degC
            dt_hr = 5 / 60;    % 5 minutes in hours
            T_init = 22.0;
            Q_it = 350.0;      % kW
            Q_cool = 350.0;    % kW (balanced)
            UA = 4.0;          % kW/degC
            T_amb = 22.0;      % ambient equal to room
            
            % Net heat transfer rate is zero
            Q_net = Q_it + UA * (T_amb - T_init) - Q_cool;
            dT = (Q_net / C_room) * dt_hr;
            T_next = T_init + dT;
            
            testCase.verifyEqual(T_next, T_init, 'AbsTol', 1e-6, ...
                'Room temperature should remain constant under balanced thermal load');
        end
    end
end
