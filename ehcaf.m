clear; clc; close all;

%% SIMULATION PARAMETERS
T = 500;
N = 12;
d = 0.5;
true_angles1 = [0, 30, 60];
Amp = 1;

%% Generate Clean Signal
s0_all = zeros(T, 1);
s_all = zeros(N-1, T);
for t = 1:T
    source = Amp * exp(1j * 2 * pi * 0.05 * (t - 1));
    s_t = zeros(N,1);
    for angle_deg = true_angles1
        steer_vec = exp(1j * 2*pi * d * (0:N-1)' * sind(angle_deg));
        s_t = s_t + steer_vec * source;
    end
    s0_all(t) = s_t(1);
    s_all(:,t) = s_t(2:end);
end

%% Add Alpha-Stable Impulsive Noise (Fresh per trial)
alpha = 1.9; beta1 = 0; c = 1; delta = 0;
noise_ref = stblrnd(alpha, beta1, c, delta, T, 1);
noise_aux = repmat(noise_ref.', N-1, 1);

s0 = s0_all + noise_ref;
s = s_all + noise_aux;

%% E-BC Complex NLMS Algorithm Implementation
% Algorithm parameters (from the image)
L = T; % Number of iterations
h = zeros(N-1, 1); % Initialize adaptive weights
w_init = 0; % w(0) = 0
sigma_e2_init = 0; % σ_e^2(0) = 0
beta = 0.9; % β parameter
lambda0 = 0.99; % λ_0 parameter
zeta = 0.01; % ζ parameter  
rho = 0.1; % ρ parameter
epsilon = 1e-6; % ε parameter (0 < ε < 1)
P = 2; % P parameter for the exponential hyperbolic cosine

% Initialize algorithm variables
w = w_init;
sigma_e2 = sigma_e2_init;

% Storage for tracking convergence
error_history = zeros(L, 1);
weights_history = zeros(N-1, L);

fprintf('Running E-BC Complex NLMS Algorithm...\n');

for l = 1:L
    % Current input samples
    x_l = s(:, l); % x(l) - auxiliary inputs (N-1 x 1)
    x0_l = s0(l); % x_0(l) - reference input
    
    % Prediction using current weights: y(l) = h^H(l)x(l)
    y_l = h' * x_l;
    
    % Error calculation: e(l) = x_0(l) - y(l)
    e_l = x0_l - y_l;
    
    % Store error for analysis
    error_history(l) = abs(e_l)^2;
    
    % Forgetting factor: λ(l) = λ_0 * e^(-|e(l)|)
    lambda_l = lambda0 * exp(-abs(e_l));
    
    % Update variance estimate: σ_e^2(l) = σ_e^2(l-1) + |1 - γ*exp(-|cosh[λ(l)e(l)]|^P)|
    gamma = 0.01; % Small positive constant
    cosh_term = cosh(lambda_l * e_l);
    sigma_e2 = sigma_e2 + abs(1 - gamma * exp(-abs(cosh_term)^P));
    
    % Regularization terms
    epsilon_r2 = 1 / sigma_e2;
    sigma_x2 = 1 / (norm(x_l)^2 + epsilon);
    
    % Step size calculation: μ'(l) = β*ε_r^2(l) + ζ*σ_x^2(l)
    mu_prime = beta * epsilon_r2 + zeta * sigma_x2;
    
    % Maximum step size: μ_max(l) = ε / (2*(l)*ρ*norm(x_l)^2)
    mu_max = epsilon / (2 * l * rho * (norm(x_l)^2 + epsilon));
    
    % Constrained step size
    if mu_prime > mu_max
        mu_l = mu_max;
    else
        mu_l = mu_prime;
    end
    
    % Exponential hyperbolic cosine derivative for robust estimation
    cosh_lambda_e = cosh(lambda_l * e_l);
    sinh_lambda_e = sinh(lambda_l * e_l);
    
    % Robust function: Δζ(l) = γλ(l)ρx*(l)exp(-|cosh[λ(l)e(l)]|^P) * 
    %                         sinh[λ(l)e(l)]|cosh[λ(l)e(l)]|^(P-1) * sign(cosh[λ(l)e(l)])
    exp_cosh_term = exp(-abs(cosh_lambda_e)^P);
    cosh_power_term = abs(cosh_lambda_e)^(P-1);
    sign_cosh = sign(real(cosh_lambda_e)); % Take real part for sign
    
    Delta_zeta = gamma * lambda_l * rho * conj(x_l) * exp_cosh_term * ...
                 sinh_lambda_e * cosh_power_term * sign_cosh;
    
    % Weight update: h(l+1) = h(l) + μ(l)Δζ(l)
    h = h + mu_l * Delta_zeta;
    
    % Store weights for analysis
    weights_history(:, l) = h;
    
    % Progress indicator
    if mod(l, 100) == 0
        fprintf('Iteration %d/%d completed\n', l, L);
    end
end

fprintf('Algorithm completed!\n');

%% Form the complete weight vector w = [1; -h]
w = [1; -h]; % (N x 1) vector

%% Spatial spectrum calculation
theta_grid = -90:1:90;
p = zeros(size(theta_grid));

fprintf('Computing spatial spectrum...\n');
for k = 1:length(theta_grid)
    theta = theta_grid(k);
    v_theta = exp(1j * 2 * pi * d * (0:N-1)' * sind(theta));
    p(k) = 1 / abs(w' * v_theta);
end

%% Plot Results
% Plot 1: Spatial spectrum
figure('Position', [100, 100, 800, 600]);
subplot(2,2,1);
plot(theta_grid, 20*log10(abs(p)), 'LineWidth', 2, 'Color', 'b');
hold on;
% Mark true angles
for angle = true_angles1
    xline(angle, '--r', sprintf('True %.0f°', angle), 'LineWidth', 1.5);
end
xlabel('Angle (degrees)');
ylabel('Spectrum (dB)');
title('Estimated Spatial Spectrum (E-BC Complex NLMS)');
grid on;
xlim([-90, 90]);

% Plot 2: Convergence of mean squared error
subplot(2,2,2);
semilogy(1:L, error_history, 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Mean Squared Error');
title('MSE Convergence');
grid on;

% % Plot 3: Weight evolution (magnitude)
% subplot(2,2,3);
% plot(1:L, abs(weights_history)', 'LineWidth', 1);
% xlabel('Iteration');
% ylabel('Weight Magnitude');
% title('Evolution of Adaptive Weights');
% grid on;
% legend(arrayfun(@(i) sprintf('h_%d', i), 1:N-1, 'UniformOutput', false), ...
%        'Location', 'best');

% % Plot 4: Final weights (real and imaginary parts)
% subplot(2,2,4);
% stem(1:N-1, real(h), 'b', 'LineWidth', 1.5, 'MarkerSize', 8);
% hold on;
% stem(1:N-1, imag(h), 'r', 'LineWidth', 1.5, 'MarkerSize', 8);
% xlabel('Weight Index');
% ylabel('Weight Value');
% title('Final Adaptive Weights');
% legend('Real Part', 'Imaginary Part');
% grid on;

%% Display Results
% fprintf('\n=== RESULTS ===\n');
% fprintf('Final adaptive weights (h):\n');
% for i = 1:length(h)
%     fprintf('h(%d) = %.4f + %.4fi\n', i, real(h(i)), imag(h(i)));
% end
% 
% fprintf('\nComplete weight vector (w = [1; -h]):\n');
% fprintf('w(1) = %.4f + %.4fi\n', real(w(1)), imag(w(1)));
% for i = 2:length(w)
%     fprintf('w(%d) = %.4f + %.4fi\n', i, real(w(i)), imag(w(i)));
% end
% 
% % Find peaks in spatial spectrum for angle estimation
% [pks, locs] = findpeaks(20*log10(abs(p)), theta_grid, 'MinPeakHeight', max(20*log10(abs(p)))-10);
% fprintf('\nEstimated angles (peaks in spectrum):\n');
% for i = 1:length(locs)
%     fprintf('Angle %d: %.1f degrees (Peak: %.2f dB)\n', i, locs(i), pks(i));
% end
% 
% fprintf('\nTrue angles: ');
% fprintf('%.0f° ', true_angles1);
% fprintf('\n');
% 
% %% Performance Metrics
% final_mse = mean(error_history(end-50:end)); % Average of last 50 iterations
% fprintf('\nFinal MSE: %.6f\n', final_mse);
% 
% % Calculate angle estimation error
% estimated_angles = locs;
% if length(estimated_angles) == length(true_angles1)
%     angle_errors = abs(sort(estimated_angles) - sort(true_angles1));
%     fprintf('Angle estimation errors: ');
%     fprintf('%.2f° ', angle_errors);
%     fprintf('\nRMSE of angle estimation: %.2f degrees\n', sqrt(mean(angle_errors.^2)));
% end