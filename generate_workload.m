function serverLoadTS = generate_workload(nDays, dtMinutes, spikeDays, spikeWindowHr, seed)
% GENERATE_WORKLOAD  Build a diurnal server-utilization timeseries with
% optional load-spike events, for driving the Server Farm block's
% "variable load" input (Q_server) in the Data Center Cooling example.
%
%   serverLoadTS = generate_workload(nDays, dtMinutes, spikeDays, spikeWindowHr, seed)
%
%   spikeDays     - 1-based day indices that get a load spike, e.g. [4 5]
%   spikeWindowHr - [startHr endHr] spike window on each spike day, e.g. [13 15]
%
%   Returns utilization in [0,1] (time in seconds). Multiply by your
%   model's rated IT power -- or map through whatever Q_server actually
%   expects -- before feeding it in. Check the units on the "Signal 1"
%   source block inside the Server Farm / Scenario subsystem.

    if nargin < 5, seed = 1; end
    rng(seed);

    dtHr = dtMinutes/60;
    t = (0:dtHr:nDays*24-dtHr)';
    hourOfDay = mod(t, 24);

    util = 0.55 + 0.20*sin(2*pi*(hourOfDay - 9)/24 - pi/2);
    util = util + 0.03*randn(size(util));          % scheduling/measurement noise
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
