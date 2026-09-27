function predictor = import_classical_surrogate(modelPath)
% IMPORT_CLASSICAL_SURROGATE Imports and wraps the trained GRU thermal surrogate for MATLAB/Simulink.
%
% Syntax:
%   predictor = import_classical_surrogate()
%   predictor = import_classical_surrogate(modelPath)
%
% Inputs:
%   modelPath - Optional path to models/classical.pt or models/classical_weights.mat
%
% Outputs:
%   predictor - Struct with fields:
%                 .predict(current_state, forecast_features) -> (horizon x 1)
%                 .update(T_room, util, tamb, pcool, carbon)
%                 .reset()
%                 .network - dlnetwork or struct containing imported weights
%
% Features:
%   - Attempts Deep Learning Toolbox importNetworkFromPyTorch('models/classical.pt')
%   - Falls back gracefully to native vectorized MATLAB GRU + MLP forward pass
%     using models/classical_weights.mat for 100% portable execution
%
% See also: mpc_controller, create_datacenter_nlmpc, run_matlab_comparison

if nargin < 1 || isempty(modelPath)
    repoRoot = fileparts(mfilename('fullpath'));
    ptPath = fullfile(repoRoot, 'models', 'classical.pt');
    matPath = fullfile(repoRoot, 'models', 'classical_weights.mat');
else
    ptPath = modelPath;
    [p, f, ~] = fileparts(modelPath);
    matPath = fullfile(p, [f, '_weights.mat']);
end

LAG_WINDOW = 12;
HORIZON = 24;

% State history buffer
history = zeros(0, 5); % [room_temp, util, tamb, pcool, carbon]

% Try importing via Deep Learning Toolbox
importedNet = [];
hasDLT = exist('importNetworkFromPyTorch', 'file') || exist('importNetworkFromPyTorch', 'builtin');

if hasDLT && exist(ptPath, 'file')
    try
        importedNet = importNetworkFromPyTorch(ptPath);
    catch
        importedNet = [];
    end
end

% Load weights for native MATLAB evaluation if needed
weights = struct();
if exist(matPath, 'file')
    weights = load(matPath);
elseif exist(ptPath, 'file')
    % If only pt exists and no mat, warn
    warning('OptimalDC:Surrogate', 'Importing PyTorch checkpoint requires Deep Learning Toolbox.');
end

% Construct Predictor Object
predictor = struct();
predictor.network = importedNet;
predictor.weights = weights;

    function update(room_temp, util, tamb, pcool, carbon)
        history(end+1, :) = [room_temp, util, tamb, pcool, carbon];
        if size(history, 1) > 200
            history = history(end-LAG_WINDOW+1:end, :);
        end
    end

    function reset()
        history = zeros(0, 5);
    end

    function preds = predict(current_state, forecast_features)
        % Predicts future temperature trajectory over HORIZON steps
        % forecast_features: (horizon x 3) [utilization, ambient_temp, carbon_intensity]
        
        if size(history, 1) < LAG_WINDOW
            preds = repmat(current_state, HORIZON, 1);
            return;
        end
        
        if isempty(fields(weights))
            preds = repmat(current_state, HORIZON, 1);
            return;
        end
        
        % Normalize inputs
        lag = history(end-LAG_WINDOW+1:end, :); % (12 x 5)
        lag_n = (lag - weights.lag_mean) ./ weights.lag_std;
        
        fc = forecast_features(1:HORIZON, :);
        fc_flat = fc(:)'; % Row vector (1 x 72)
        fc_n = (fc_flat - weights.fc_mean) ./ weights.fc_std;
        
        % Forward pass through GRU + MLP
        % 1. MLP branch for forecast
        % fc_net: Linear(72, 64) -> ReLU
        h_fc = max(0, fc_n * weights.fc_net_0_weight' + weights.fc_net_0_bias);
        
        % 2. GRU recurrent forward pass over lag window
        % For 2-layer GRU with 64 hidden units
        h1 = zeros(1, 64);
        h2 = zeros(1, 64);
        for t = 1:LAG_WINDOW
            xt = lag_n(t, :);
            % Layer 1 GRU step
            h1 = gru_cell_step(xt, h1, ...
                weights.gru_weight_ih_l0, weights.gru_weight_hh_l0, ...
                weights.gru_bias_ih_l0, weights.gru_bias_hh_l0);
            % Layer 2 GRU step
            h2 = gru_cell_step(h1, h2, ...
                weights.gru_weight_ih_l1, weights.gru_weight_hh_l1, ...
                weights.gru_bias_ih_l1, weights.gru_bias_hh_l1);
        end
        h_gru = h2;
        
        % 3. Output MLP: Linear(128, 64) -> ReLU -> Linear(64, 24)
        h_cat = [h_gru, h_fc];
        h_out1 = max(0, h_cat * weights.out_net_0_weight' + weights.out_net_0_bias);
        delta_norm = h_out1 * weights.out_net_2_weight' + weights.out_net_2_bias;
        
        % Denormalize target delta
        delta = delta_norm * weights.y_std + weights.y_mean;
        preds = current_state + delta(:);
    end

predictor.update = @update;
predictor.reset = @reset;
predictor.predict = @predict;

end

% Local GRU cell computation matching PyTorch formulation
function ht = gru_cell_step(x, h_prev, W_ih, W_hh, b_ih, b_hh)
% PyTorch GRU equations:
% r_t = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
% z_t = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
% n_t = tanh(W_in * x + b_in + r_t * (W_hn * h + b_hn))
% h_t = (1 - z_t) * n_t + z_t * h_prev

hidden_size = 64;
gi = x * W_ih' + b_ih;
gh = h_prev * W_hh' + b_hh;

% Slices for [r, z, n] gates
r = 1.0 ./ (1.0 + exp(-(gi(1:hidden_size) + gh(1:hidden_size))));
z = 1.0 ./ (1.0 + exp(-(gi(hidden_size+1:2*hidden_size) + gh(hidden_size+1:2*hidden_size))));
n = tanh(gi(2*hidden_size+1:3*hidden_size) + r .* gh(2*hidden_size+1:3*hidden_size));

ht = (1.0 - z) .* n + z .* h_prev;
end
