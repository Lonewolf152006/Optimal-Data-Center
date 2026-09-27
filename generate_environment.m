function ambientTempTS = generate_environment(nDays, dtMinutes, heatWaveDays, seed)
% GENERATE_ENVIRONMENT  Build a diurnal ambient-temperature timeseries with
% optional heat-wave days, for driving the "variable environment" input of
% the Data Center Cooling Simscape example.
%
%   ambientTempTS = generate_environment(nDays)
%   ambientTempTS = generate_environment(nDays, dtMinutes, heatWaveDays, seed)
%
%   nDays        - number of days to simulate (e.g. 5)
%   dtMinutes    - sample interval in minutes (default: 5)
%   heatWaveDays - vector of 1-based day indices that get a +8 degC heat
%                  wave applied all day, e.g. [3 5]. Pass [] for none.
%                  (default: [])
%   seed         - RNG seed for reproducibility (default: 1)
%
%   Returns a MATLAB timeseries in degC (time in seconds), ready to feed a
%   "From Workspace" block, or assign to whatever workspace variable your
%   Environment/Scenario subsystem's "variable" variant expects.
%
%   NOTE: this is an illustrative synthetic profile, not measured weather
%   data. Replace with real TMY3 / local weather-station data for a
%   production analysis -- the shape (diurnal + occasional heat wave) is
%   what matters for validating the control logic.

    arguments
        nDays (1,1) {mustBePositive, mustBeInteger}
        dtMinutes (1,1) {mustBePositive} = 5
        heatWaveDays (1,:) double = []
        seed (1,1) {mustBeNumeric} = 1
    end

    % Secondary nargin guards (belt-and-suspenders)
    if nargin < 2 || isempty(dtMinutes), dtMinutes = 5; end
    if nargin < 3, heatWaveDays = []; end
    if nargin < 4, seed = 1; end
    rng(seed);

    % Guard: clamp heatWaveDays to valid day range
    if ~isempty(heatWaveDays)
        heatWaveDays = heatWaveDays(heatWaveDays >= 1 & heatWaveDays <= nDays);
    end

    dtHr = dtMinutes/60;
    t = (0:dtHr:nDays*24-dtHr)';          % hours
    hourOfDay = mod(t, 24);

    % Diurnal sinusoid: trough ~05:00, peak ~15:00
    Tamb = 24.0 + 6.5*sin(2*pi*(hourOfDay - 9)/24 - pi/2);

    % Small stochastic weather noise so successive runs aren't identical
    Tamb = Tamb + 0.5*randn(size(Tamb));

    if ~isempty(heatWaveDays)
        dayIdx = floor(t/24) + 1;         % 1-based day number
        isHeatWave = ismember(dayIdx, heatWaveDays);
        Tamb = Tamb + 8.0*isHeatWave;
    end

    ambientTempTS = timeseries(Tamb, t*3600, 'Name', 'AmbientTemp');
    ambientTempTS.DataInfo.Units = 'degC';
end
