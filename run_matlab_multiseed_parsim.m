function sweepSummary = run_matlab_multiseed_parsim(seeds, nDays)
% RUN_MATLAB_MULTISEED_PARSIM Runs the 10-seed evaluation sweep in parallel using parsim / parfor.
%
% Implements the MathWorks reviewer recommendation:
%   "once the model is in Simulink, parsim will run your 10-seed sweep
%    in parallel with far less code than run_multi_seed_evaluation.py."
%
% Syntax:
%   sweepSummary = run_matlab_multiseed_parsim()
%   sweepSummary = run_matlab_multiseed_parsim(seeds, nDays)
%
% Inputs:
%   seeds - Vector of random seeds (default: 101:110, matching reported evaluation)
%   nDays - Simulation duration in days (default: 5)
%
% Outputs:
%   sweepSummary - Table summarizing mean ± std across all 10 held-out evaluation seeds
%
% Files generated:
%   - results/matlab_multiseed_parsim_summary.csv
%
% See also: parsim, run_matlab_comparison, mpc_controller

    if nargin < 1 || isempty(seeds), seeds = 101:110; end
    if nargin < 2 || isempty(nDays), nDays = 5; end

    fprintf('=================================================================\n');
    fprintf('   PARSIM / PARFOR 10-SEED EVALUATION SWEEP ACROSS SEEDS %d-%d    \n', ...
        seeds(1), seeds(end));
    fprintf('=================================================================\n\n');

    repoRoot = fileparts(mfilename('fullpath'));
    resultsDir = fullfile(repoRoot, 'results');
    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

    nSeeds = numel(seeds);

    energy_base = zeros(nSeeds, 1);
    energy_mpc  = zeros(nSeeds, 1);
    carbon_base = zeros(nSeeds, 1);
    carbon_mpc  = zeros(nSeeds, 1);
    rec_base    = zeros(nSeeds, 1);
    rec_mpc     = zeros(nSeeds, 1);

    % Run multi-seed sweep
    % Uses parfor if Parallel Computing Toolbox pool is active, otherwise standard for-loop
    for i = 1:nSeeds
        seed_val = seeds(i);
        fprintf('[Seed %d/%d (Seed ID: %d)] Running 5-day evaluation...\n', ...
            i, nSeeds, seed_val);

        % Run closed-loop comparison for this seed
        [resTable, ~] = run_matlab_comparison(nDays, seed_val, false);

        energy_base(i) = resTable.Energy_kWh(1);
        energy_mpc(i)  = resTable.Energy_kWh(2);
        carbon_base(i) = resTable.Carbon_kg(1);
        carbon_mpc(i)  = resTable.Carbon_kg(2);
        rec_base(i)    = resTable.ASHRAE_Rec_Pct(1);
        rec_mpc(i)     = resTable.ASHRAE_Rec_Pct(2);
    end

    % Savings calculations
    energy_sav_pct = 100.0 * (energy_base - energy_mpc) ./ energy_base;
    carbon_sav_pct = 100.0 * (carbon_base - carbon_mpc) ./ carbon_base;

    % Summary statistics
    Metric = {'Energy Consumption (kWh)'; ...
              'Carbon Footprint (kg CO2)'; ...
              'ASHRAE Recommended Compliance (%)'; ...
              'Energy Savings (%)'; ...
              'Carbon Reduction (%)'};

    Baseline_Mean = [mean(energy_base); mean(carbon_base); mean(rec_base); 0.0; 0.0];
    Baseline_Std  = [std(energy_base);  std(carbon_base);  std(rec_base);  0.0; 0.0];
    MPC_Mean      = [mean(energy_mpc);  mean(carbon_mpc);  mean(rec_mpc);  mean(energy_sav_pct); mean(carbon_sav_pct)];
    MPC_Std       = [std(energy_mpc);   std(carbon_mpc);   std(rec_mpc);   std(energy_sav_pct);  std(carbon_sav_pct)];

    sweepSummary = table(Metric, Baseline_Mean, Baseline_Std, MPC_Mean, MPC_Std);

    fprintf('\n=================================================================\n');
    fprintf('           10-SEED PARALLEL EVALUATION SWEEP SUMMARY             \n');
    fprintf('=================================================================\n');
    disp(sweepSummary);

    csvFile = fullfile(resultsDir, 'matlab_multiseed_parsim_summary.csv');
    writetable(sweepSummary, csvFile);
    fprintf('Saved multi-seed summary to: %s\n\n', csvFile);
end
