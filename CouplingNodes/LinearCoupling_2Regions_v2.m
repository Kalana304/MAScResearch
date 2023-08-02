% clc; clear; close all;

%% Define Parameters

T = 2500;       % Time of simulation in msec

Trials = 1;     % No. of trials (if averaging is needed)

Ne = 800;       % No. of excitatory units
Ni = 200;       % No. of inhibitory units
dt = 0.1;       % Time step for integration

temp_window = 20;   % Temporal window for spike coherence measures

global beta gamma threshold

beta = 300;         % activation function non-linear gain
gamma = 0.016;      % scaling to mV
threshold = 0.0;    % activation function threshold (inflexion point)

D0 = 0.001;     % noise variance

I_e0 = -0.25;   % Bias current e cells (Add current here to trigger seizure like state/gamma)
I_i0 = -0.5;    % Bias current i cells

sigma_e = [4.4, 4.4]';  % heterogeneity in e cells
sigma_i = [2.5, 2.5]';  % heterogeneity in i cells

w_ee0 = 1.6;    % e --> e synaptic strength
w_ie0 = -4.7;   % i --> e synaptic strength
w_ei0 = 3.0;    % e --> i synaptic strength
w_ii0 = -0.13;  % i --> i synaptic strength

alpha_e = [1, 1]';        % time scale E units
alpha_i = [2, 2]';        % time scale I units

b = 0.0;              % adaptation gain
alpha_adapt = 0.01; % adaptation time scale - should be slower

tVect = 0 : T; Nt = length(tVect);

%% Network creation

w_ee = w_ee0 / gamma; % mV
w_ei = w_ei0 / gamma; % mV
w_ie = w_ie0 / gamma; % mV
w_ii = w_ii0 / gamma; % mV

D = D0 / (gamma^2); 
I_i = I_i0 / gamma;
I_e = I_e0 / gamma; 

nRegions = length(sigma_i);

sigmas = zeros(nRegions, 2);
sigmas(:, 1) = sigma_e'; sigmas(:, 2) = sigma_i';
fprintf('sigma_e = %.3f | sigma_i = %.3f \n', sigmas');

%% Solving the ODEs using Euler Method

Ue_trial = zeros(nRegions, Nt); Ui_trial = zeros(nRegions, Nt);
Fe_trial = zeros(nRegions, Nt);

% Global Coupling coefficient - homogenous coupling
C = 0.0;

rng('default'); a_12 = 1 / 2; a_21 = 1 / 2;
A = zeros(nRegions, nRegions); A(1, 2) = a_12; A(2, 1) = a_21;

for q = 1 : Trials
    % Define Noise Arrays
    rng(q * 187198 + 56 * randi(105) + 1); Xi_e = randn(nRegions, Nt); 
    rng(3 + q * 8976); Xi_i = randn(nRegions, Nt);
    
    rng('default');
    % Initialization 
    U_e = randn(nRegions, Nt); U_i = randn(nRegions, Nt);
    V_e = randn(nRegions, Nt); V_i = randn(nRegions, Nt);
    Fe = zeros(nRegions, Nt); 
    
    for t = 1 : T
        I_t = [I_e + (t / T) * 0.5 / gamma, I_e]';
        U_e(:, t + 1) = U_e(:, t) + dt * alpha_e .* (-U_e(:, t) + b * V_e(:, t) + ...
                        w_ee * F(U_e(:, t), sigma_e) + C * A * U_e(:, t) + ...
                        w_ie * F(U_i(:, t), sigma_i) + I_t) + ...
                        sqrt(2 * alpha_e * D * dt) .* Xi_e(:, t) ./ Ne;
        
        Fe(:, t) = F(U_e(:, t) + C * A * U_e(:, t), sigma_e);

        U_i(:, t + 1) = U_i(:, t) + dt * alpha_i .* (-U_i(:, t) + b * V_i(:, t) + ...
                        w_ei * F(U_e(:, t), sigma_e) + ...
                        w_ii * F(U_i(:, t), sigma_i) + I_i) + ...
                        sqrt(2 * alpha_i * D * dt) .* Xi_i(:, t) ./ Ni;

        V_e(:,t+1) = V_e(:, t) + dt * alpha_adapt * (-V_e(:, t) + U_e(:, t) - I_e);
        V_i(:, t+1) = V_i(:, t) + dt * alpha_adapt * (-V_i(:, t) + U_i(:, t) - I_i);
    end

    Ue_trial = Ue_trial + U_e;
    Ui_trial = Ui_trial + U_i;
    Fe_trial = Fe_trial + Fe;
end

Ue_trial = Ue_trial / Trials; Ui_trial = Ui_trial / Trials;
Fe_trial = Fe_trial / Trials;

figure(1)
subplot(nRegions, 2, 1); plot(tVect, Ue_trial(1, :), 'LineWidth', 1.2); xlabel('time (ms)'); ylabel('potential (mV)')
title(sprintf('Dynamics of an Excitatory Population\n%c_e = %.2f and %c_i = %.2f with C = %.2f', 963, sigma_e(1), 963, sigma_i(1), C))
subplot(nRegions, 2, 2); plot(tVect, Ui_trial(1,:), 'LineWidth', 1.2); xlabel('time (ms)'); ylabel('potential (mV)')
title(sprintf('Dynamics of an Inihibitory Population\n%c_e = %.2f and %c_i = %.2f with C = %.2f', 963, sigma_e(1), 963, sigma_i(1), C))
subplot(nRegions, 2, 3); plot(tVect, Ue_trial(2, :), 'LineWidth', 1.2); xlabel('time (ms)'); ylabel('potential (mV)')
title(sprintf('%c_e = %.2f and %c_i = %.2f', 963, sigma_e(2), 963, sigma_i(2)))
subplot(nRegions, 2, 4); plot(tVect, Ui_trial(2,:), 'LineWidth', 1.2); xlabel('time (ms)'); ylabel('potential (mV)')
title(sprintf('%c_e = %.2f and %c_i = %.2f', 963, sigma_e(2), 963, sigma_i(2)))

figure(2)
subplot(1, 2, 1);
plot(Ue_trial(1, :), Fe_trial(1, :))
subplot(1, 2, 2);
plot(Ue_trial(2, :), Fe_trial(2, :))

figure(3)
subplot(1, 2, 1);
plot(Ue_trial(1, :), Ui_trial(1, :), 'LineWidth', 1.2); 
xlabel('U_e'); ylabel('U_i');
title('Region 01 - Phase Plot')
subplot(1, 2, 2);
plot(Ue_trial(2, :), Ui_trial(2, :), 'LineWidth', 1.2)
xlabel('U_e'); ylabel('U_i');
title('Region 02 - Phase Plot')

% Mean-field firing fucntion
function U_bar = F(U, sigma)
    global threshold gamma beta

    % parameters for convolution sum
    v_min = -1 / gamma; v_max = 1 / gamma; nConv = 1000;
    dv = abs(v_max - v_min) / nConv;

    % Firing rate function
    f = @(u, theta) 1 ./ (1 + exp(-beta * gamma * (u - theta)));
        
    U_bar = zeros(length(U), 1);
    
    for r = 1 : length(U)
        for i = 1 : nConv
            v = v_min + (i-1) * dv;
            U_bar(r) = U_bar(r) + dv * f(U(r) + v, threshold) *... 
                    exp(-v^2 / (2 * sigma(r)^2)) / sqrt(2 * pi * sigma(r)^2);   
        end   
    end
end