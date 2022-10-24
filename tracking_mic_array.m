clear all
close all

addpath("./project")

% Lock seed
randn('state',123);

dt = 1024/48000;       % Sampling period

% Implement RMSE (true and estimate)
rmse = @(X,EST) sqrt(mean(sum((X-EST).^2)));

%% Try with scene
load('shmicsigs/scene2.mat', 'shmovsig', 'fs', 'src_xyz', 'rec_xyz', 'motion_xyz', 'room')
sig1 = shmovsig(:,:,1);
sig2 = shmovsig(:,:,2);
sig3 = shmovsig(:,:,3);

diff_th = 0.15;
diff_mult = 10;

% samples per step
smpStep = dt*fs;
numSteps = floor(size(sig1, 1) / smpStep);
smpIdx = 1;

receivers = rec_xyz;

% True Trajectory
X = zeros(3, numSteps);
X(1, :) = interp1([1, numSteps/2, numSteps], motion_xyz(:,1), [1:numSteps]);
X(2, :) = interp1([1, numSteps/2, numSteps], motion_xyz(:,2), [1:numSteps]);
X(3, :) = interp1([1, numSteps/2, numSteps], motion_xyz(:,3), [1:numSteps]);


%% Dynamic model

% Parameters of the dynamic model
qc = 1;

% This is the transition matrix
A  = [1 0 0 dt 0 0;
      0 1 0 0 dt 0;
      0 0 1 0 0 dt;
      0 0 0 1 0 0;
      0 0 0 0 1 0;
      0 0 0 0 0 1];

% This is the process noise covariance
Q = [qc*dt^3/3 0 0 qc*dt^2/2 0 0;
    0 qc*dt^3/3 0 0 qc*dt^2/2 0;
    0 0 qc*dt^3/3 0 0 qc*dt^2/2;
    qc*dt^2/2 0 0 qc*dt 0 0;
    0 qc*dt^2/2 0 0 qc*dt 0;
    0 0 qc*dt^2/2 0 0 qc*dt];


% Measurements
sd = deg2rad(5);       % Standard deviation of measurements
R  = sd^2*eye(6);       % The joint covariance

% Initializations
x0 = [X(:,1);0;0;0];     % Initial state
x0 = x0 + [randn(3,1);0;0;0];  % little error on first
P0 = eye(6);            % Some uncertainty


%% Parameter estimator
S1 = receivers(1, :).';
S2 = receivers(2, :).';
S3 = receivers(3, :).';

ACN2WXYZ = sqrt(4*pi) * [1 0 0 0 ;
                    0 0 0 1/sqrt(3);
                    0 1/sqrt(3) 0 0;
                    0 0 1/sqrt(3) 0];

Y = [];
Y_diff = [];
for k=1:numSteps
    blockS1 = sig1(smpIdx:smpIdx+smpStep-1, :);
    blockS2 = sig2(smpIdx:smpIdx+smpStep-1, :);
    blockS3 = sig3(smpIdx:smpIdx+smpStep-1, :);
    
    % convert to B-format
    blockS1 = ACN2WXYZ * blockS1.';
    blockS2 = ACN2WXYZ * blockS2.';
    blockS3 = ACN2WXYZ * blockS3.';
    
    % extract measurements
    [intensity1, energy1, diff1] = estimateParameters(blockS1);
    [intensity2, energy2, diff2] = estimateParameters(blockS2);
    [intensity3, energy3, diff3] = estimateParameters(blockS3);
    
    intensity1 = mean(intensity1, 2);
    intensity2 = mean(intensity2, 2);
    intensity3 = mean(intensity3, 2);
    
    diff1 = mean(diff1, 2);
    diff2 = mean(diff2, 2);
    diff3 = mean(diff3, 2);
    
    [azi1, ele1, r] = cart2sph(intensity1(1), intensity1(2), intensity1(3));
    [azi2, ele2, r] = cart2sph(intensity2(1), intensity2(2), intensity2(3));
    [azi3, ele3, r] = cart2sph(intensity3(1), intensity3(2), intensity3(3));
    
    noisey = [azi1; ele1; azi2; ele2; azi3; ele3];
    Y = [Y noisey];
    Y_diff = [Y_diff [diff1; diff2; diff3]];
    
    smpIdx = smpIdx+smpStep;
end


%% Intersection (Baseline-Approach)
mk = x0;                % Initialize
Pk = P0;                %

EST_ISC = zeros(6,numSteps);
P_ISC = zeros(6,numSteps);


for k=1:numSteps
    
    Y_k = Y(:, k);  % Current measurement
    p_isc1 = doaIntersect(S1, Y_k(1:2), S2, Y_k(3:4));
    p_isc2 = doaIntersect(S1, Y_k(1:2), S3, Y_k(5:6));
    p_isc3 = doaIntersect(S2, Y_k(3:4), S3, Y_k(5:6));

    p_isc_avg = 1/3*sum(cat(2, p_isc1, p_isc2, p_isc3), 2);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~any((isnan(p_isc_avg)))
        mk = cat(1, p_isc_avg, zeros(3, 1));
    end
    
    EST_ISC(:,k) = mk;
    %P_ISC(:,k) = diag(Pk);
    
end


% Compute error
err_ISC = rmse(X(1:3, :),EST_ISC(1:3, :))

% Plot
figure;
plot3(X(1,:),X(2,:),X(3,:),'--',...
      EST_ISC(1,:),EST_ISC(2,:),EST_ISC(3,:),'-',...
      S1(1),S1(2),S1(3),'x',...
      S2(1),S2(2),S2(3),'x',...
      S3(1),S3(2),S3(3),'x',...
      'LineWidth',2, 'MarkerSize',10)
legend('True trajectory','Mean ISC estimate','Sensor 1','Sensor 2', 'Sensor 3');
title('\bf Mean Intersection Solution')

xlabel('x'); ylabel('y'), zlabel('z')
axis equal
grid on
view(3)
%exportgraphics(gcf,'./figs/Real_UKF.pdf')




%% UKF - naive
mk = x0;                % Initialize
Pk = P0;                %


EST_UKF = zeros(6,numSteps);
P_UKF = zeros(6,numSteps);

% Setup sigma points and weights
alpha = 0.9;
kappa = 3-6;
lambda = @(n) alpha^2 * (n+kappa) - n;
beta = 2;
W_m = @(n) [lambda(n)/(n+lambda(n)), repelem(1/(2*(n+(lambda(n)))), 2*n)];
W_c = @(n) [lambda(n)/(n+lambda(n)) + 1-alpha^2+beta, repelem(1/(2*(n+(lambda(n)))), 2*n)];

n = length(mk);
nu = sqrt(n+lambda(n));
W_m_ = W_m(n);
W_c_ = W_c(n);

for k=1:numSteps
    % Prediction (linear)
    mp = A * mk;
    Pp = A * Pk * A.' + Q;
    
    % Update
    % Recalculation of sigma points
    sqrtP = chol(Pp, 'lower');
    X_sp = [mp, mp + nu*sqrtP, mp - nu*sqrtP];
    % Propagate
    Y_sp = measurementFunction(X_sp, [S1,S2,S3]);
    
    % Problem here : azi wrapping
    mu = sum(W_m(n)  .* Y_sp, 2);
    S = W_c(n) .* (Y_sp - mu) * (Y_sp - mu).' + R;
    C = W_c(n) .* (X_sp - mp) * (Y_sp - mu).';

    % Filter
    K = C/S;
    
    Y_k = Y(:, k);  % Current measurement
    v = Y_k - mu;
    % As well as wrapping here:
    mk = mp + K*v;
    Pk = Pp - K*S*K';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    EST_UKF(:,k) = mk;
    P_UKF(:,k) = diag(Pk);
    
end

% Compute error
err_UKF = rmse(X(1:3, :),EST_UKF(1:3, :))

% Plot
figure;
plot3(X(1,:),X(2,:),X(3,:),'--',...
      EST_UKF(1,:),EST_UKF(2,:),EST_UKF(3,:),'-',...
      S1(1),S1(2),S1(3),'x',...
      S2(1),S2(2),S2(3),'x',...
      S3(1),S3(2),S3(3),'x',...
      'LineWidth',2, 'MarkerSize',10)
legend('True trajectory','UKF estimate','Sensor 1','Sensor 2', 'Sensor 3');
title('\bf UKF Solution')

xlabel('x'); ylabel('y'), zlabel('z')
axis equal
grid on
view(3)
%exportgraphics(gcf,'./figs/Real_UKF.pdf')

figure; hold on
plot(linspace(0, numSteps*dt, numSteps), P_UKF(1, :).', ...
     linspace(0, numSteps*dt, numSteps), P_UKF(2, :).', ...
     linspace(0, numSteps*dt, numSteps), P_UKF(3, :).')

plot(linspace(0, numSteps*dt, numSteps), P_UKF(4, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_UKF(5, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_UKF(6, :).', '--')
 
legend('x', 'y', 'z', 'dx', 'dy', 'dz')
xlabel('t in s')
title("P UKF")


%% UKF Sph
mk = x0;                % Initialize
Pk = P0;                %

EST_UKF_SPH = zeros(6,numSteps);
P_UKF_SPH = zeros(6,numSteps);

% Track diffuseness condition for this one
triggered_c = zeros(size(Y_diff));

% Setup sigma points and weights
alpha = 0.9;
kappa = 3-6;
lambda = @(n) alpha^2 * (n+kappa) - n;
beta = 2;
W_m = @(n) [lambda(n)/(n+lambda(n)), repelem(1/(2*(n+(lambda(n)))), 2*n)];
W_c = @(n) [lambda(n)/(n+lambda(n)) + 1-alpha^2+beta, repelem(1/(2*(n+(lambda(n)))), 2*n)];

n = length(mk);
nu = sqrt(n+lambda(n));
W_m_ = W_m(n);
W_c_ = W_c(n);

for k=1:numSteps
    % Prediction (linear)
    mp = A * mk;
    Pp = A * Pk * A.' + Q;
    
    % Update
    % Recalculation of sigma points
    sqrtP = chol(Pp, 'lower');
    X_sp = [mp, mp + nu*sqrtP, mp - nu*sqrtP];
    % Propagate
    Y_sp = measurementFunction(X_sp, [S1,S2,S3]);
    
    % Circular mean and Cov
    mu = circularMean(Y_sp, W_m_);
    S = W_c_ .* angle(exp(1i*(Y_sp-mu))) * (angle(exp(1i*(Y_sp-mu))))' + R;
    C = W_c_ .* (X_sp-mp) * (angle(exp(1i*(Y_sp-mu)))).';
    
    % Regularize sensor data over threshold
    ovrTh = repelem(Y_diff(:, k) > diff_th, 2);  
    if any(ovrTh)
        triggered_c(ovrTh(1:2:end),k) = (Y_diff(ovrTh(1:2:end), k));
        Rp = zeros(size(R));
        Rp(:, ovrTh) = R(:, ovrTh) .* (repelem(Y_diff(:, k), 2))*diff_mult;
        S = S + Rp;
    end

    % Filter
    K = C/S;

    Y_k = Y(:, k);  % Current measurement
    % And here
    v = angle(exp(1i*(Y_k-mu)));
    mk = mp + K*v;
    Pk = Pp - K*S*K';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    EST_UKF_SPH(:,k) = mk;
    P_UKF_SPH(:,k) = diag(Pk);
    
end

% Compute error
err_UKF_SPH = rmse(X(1:3, :),EST_UKF_SPH(1:3, :))

% Plot
figure;
plot3(X(1,:),X(2,:),X(3,:),'--',...
      EST_UKF_SPH(1,:),EST_UKF_SPH(2,:),EST_UKF_SPH(3,:),'-',...
      S1(1),S1(2),S1(3),'x',...
      S2(1),S2(2),S2(3),'x',...
      S3(1),S3(2),S3(3),'x',...
      'LineWidth',2, 'MarkerSize',10)
legend('True trajectory','UKF estimate','Sensor 1','Sensor 2', 'Sensor 3');
title('\bf UKF SPH Solution')

xlabel('x'); ylabel('y'), zlabel('z')
axis equal
grid on
view(3)
%exportgraphics(gcf,'./figs/Real_UKF.pdf')

figure; hold on
plot(linspace(0, numSteps*dt, numSteps), P_UKF_SPH(1, :).', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_SPH(2, :).', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_SPH(3, :).')

plot(linspace(0, numSteps*dt, numSteps), P_UKF_SPH(4, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_SPH(5, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_SPH(6, :).', '--')
 
legend('x', 'y', 'z', 'dx', 'dy', 'dz')
xlabel('t in s')
title("P UKF SPH")



%% Square Root Unscented Kalman Filter - SPH
% mk = x0;                % Initialize
% Pk = P0;                %
% 
% EST_SRUKF = zeros(6,numSteps);
% P_SRUKF = zeros(6,numSteps);
% 
% 
% Sk = chol(Pk, 'lower');
% sqrtQ = chol(Q, 'lower');
% sqrtR = chol(R, 'lower');
% 
% for k=1:numSteps
%     % Prediction
%     sp = [mk, mk + nu*Sk, mk - nu*Sk];
%     X_sp = A * sp;
%     mp = sum(W_m_ .* X_sp, 2);  % could write directly mp = A * mk;
%     
%     assert(all(W_c_ >= 0))
%     [~,Sp] = qr([sqrt(W_c_(2:end)) .* (X_sp(:,2:end) - mp) , sqrtQ].', 0);
%     Sp = cholupdate(Sp, W_c_(1) * (X_sp(:, 1) - mp));
% 
%     % Update
%     Y_sp = measurementFunction(X_sp, [S1,S2,S3]);
%     mu = circularMean(Y_sp, W_m_);
%     C = W_c_ .* (X_sp-mp) * (angle(exp(1i*(Y_sp-mu)))).';
%     
%     assert(all(W_c_ >= 0))
%     [~,Sy] = qr([sqrt(W_c_(2:end)) .* angle(exp(1i*(Y_sp(:,2:end) - mu))), ...
%                  sqrtR].', 0);
%     Sy = cholupdate(Sy, W_c_(1) * angle(exp(1i*(Y_sp(:,1) - mu))));
%     
% 
%     % Regularize sensor data over threshold
%     ovrTh = repelem(Y_diff(:, k) > diff_th, 2);  
%     if any(ovrTh)
%        Rp = zeros(size(R));
%        Rp(:, ovrTh) = sqrtR(:, ovrTh) .* (repelem(Y_diff(:, k), 2))*diff_mult;
%        Sy = Sy + Rp;
%     end
%     % Filter
%     K = (C/Sy.')/Sy;    
% 
%     U = K*Sy;
%     Sk = Sp;
%     for itm = 1:n
%         [Sk, err] = cholupdate(Sk, U(:, itm), '-');
%         if(err)
%             warning('CholUpdate failed at k=%i', k);
%             break
%         end
%     end
% 
%     
%     Y_k = Y(:, k);  % Current measurement
%     %v = Y_k - mu;
%     %v = mod((v + pi/2), pi) - pi/2;
%     v = angle(exp(1i*(Y_k-mu)));
%     mk = mp + K*v;
%     Pk = Sk.' * Sk;
%     
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%     EST_SRUKF(:,k) = mk;
%     P_SRUKF(:,k) = diag(Pk);
%     
% end
% 
% 
% % Compute error
% err_SRUKF = rmse(X(1:3, :),EST_SRUKF(1:3, :))
% 
% % Plot
% figure;
% plot3(X(1,:),X(2,:),X(3,:),'--',...
%       EST_SRUKF(1,:),EST_SRUKF(2,:),EST_SRUKF(3,:),'-',...
%       S1(1),S1(2),S1(3),'x',...
%       S2(1),S2(2),S2(3),'x',...
%       S3(1),S3(2),S3(3),'x',...
%       'LineWidth',2, 'MarkerSize',10)
% legend('True trajectory','UKF estimate','Sensor 1','Sensor 2', 'Sensor 3');
% title('\bf SRUKF - SPH Solution')
% 
% xlabel('x'); ylabel('y'), zlabel('z')
% axis equal
% grid on
% view(3)
% %exportgraphics(gcf,'./figs/Real_UKF.pdf')
% 
% figure; hold on
% plot(linspace(0, numSteps*dt, numSteps), P_SRUKF(1, :).', ...
%      linspace(0, numSteps*dt, numSteps), P_SRUKF(2, :).', ...
%      linspace(0, numSteps*dt, numSteps), P_SRUKF(3, :).')
% 
% plot(linspace(0, numSteps*dt, numSteps), P_SRUKF(4, :).', '--', ...
%      linspace(0, numSteps*dt, numSteps), P_SRUKF(5, :).', '--', ...
%      linspace(0, numSteps*dt, numSteps), P_SRUKF(6, :).', '--')
%  
% legend('x', 'y', 'z', 'dx', 'dy', 'dz')
% xlabel('t in s')
% title("P SRUKF - SPH")


%% EKF Sph
mk = x0;                % Initialize
Pk = P0;                %

EST_EKF_SPH = zeros(6,numSteps);
P_EKF_SPH = zeros(6,numSteps);

% Jacobian
% for azi
Hx1 = @(x, y, z, sx, sy, sz) (1 + ((y-sy)/(x-sx)).^2).^(-1).*(-1*(y-sy)./(x-sx).^2);
Hy1 = @(x, y, z, sx, sy, sz) (1 + ((y-sy)/(x-sx)).^2).^(-1).*1./(x-sx);
Hz1 = @(x, y, z, sx, sy, sz) 0;

% for ele
Hx2 = @(x, y, z, sx, sy, sz) (1 + ((z-sz)/sqrt((x-sx).^2 + ((y-sy).^2))).^2).^(-1)...
    .*(-(x-sx).*(z-sz)./((x-sx).^2 + (y-sy).^2).^(3/2));
Hy2 = @(x, y, z, sx, sy, sz) (1 + ((z-sz)/sqrt((x-sx).^2 + ((y-sy).^2))).^2).^(-1)...
    .*(-(y-sy).*(z-sz)./((x-sx).^2 + (y-sy).^2).^(3/2));
Hz2 = @(x, y, z, sx, sy, sz) (1 + ((z-sz)/sqrt((x-sx).^2 + ((y-sy).^2))).^2).^(-1)...
    .*(1./((x-sx).^2 + (y-sy).^2).^(1/2));
H = @(x) [

    Hx1(x(1), x(2), x(3), S1(1), S1(2), S1(3)), Hy1(x(1), x(2), x(3), S1(1), S1(2), S1(3)), Hz1(x(1), x(2), x(3), S1(1), S1(2), S1(3)), zeros(1, 3);
    Hx2(x(1), x(2), x(3), S1(1), S1(2), S1(3)), Hy2(x(1), x(2), x(3), S1(1), S1(2), S1(3)), Hz2(x(1), x(2), x(3), S1(1), S1(2), S1(3)), zeros(1, 3);

    Hx1(x(1), x(2), x(3), S2(1), S2(2), S2(3)), Hy1(x(1), x(2), x(3), S2(1), S2(2), S2(3)), Hz1(x(1), x(2), x(3), S2(1), S2(2), S2(3)), zeros(1, 3);
    Hx2(x(1), x(2), x(3), S2(1), S2(2), S2(3)), Hy2(x(1), x(2), x(3), S2(1), S2(2), S2(3)), Hz2(x(1), x(2), x(3), S2(1), S2(2), S2(3)), zeros(1, 3);

    Hx1(x(1), x(2), x(3), S3(1), S3(2), S3(3)), Hy1(x(1), x(2), x(3), S3(1), S3(2), S3(3)), Hz1(x(1), x(2), x(3), S3(1), S3(2), S3(3)), zeros(1, 3);
    Hx2(x(1), x(2), x(3), S3(1), S3(2), S3(3)), Hy2(x(1), x(2), x(3), S3(1), S3(2), S3(3)), Hz2(x(1), x(2), x(3), S3(1), S3(2), S3(3)), zeros(1, 3);

];


for k=1:numSteps
    % Prediction (No linearization necessary, this is accurate)
    mp = A*mk;
    Pp = A*Pk*A.' + Q;


    % Update
    % Propagate
    Y_p = measurementFunction(mp, [S1,S2,S3]);
    S = H(mp)*Pp*H(mp)' + R;
    
    
    % Regularize sensor data over threshold
    ovrTh = repelem(Y_diff(:, k) > diff_th, 2);  
    if any(ovrTh)
        Rp = zeros(size(R));
        Rp(:, ovrTh) = R(:, ovrTh) .* (repelem(Y_diff(:, k), 2))*diff_mult;
        S = S + Rp;
    end

    % Filter
    K = (Pp*H(mp)')/S;

    Y_k = Y(:, k);  % Current measurement
    % And here
    v = angle(exp(1i*(Y_k-Y_p)));
    
    mk = mp + K*v;
    Pk = Pp - K*S*K';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    EST_EKF_SPH(:,k) = mk;
    P_EKF_SPH(:,k) = diag(Pk);
    
end

% Compute error
err_EKF_SPH = rmse(X(1:3, :),EST_EKF_SPH(1:3, :))

% Plot
figure;
plot3(X(1,:),X(2,:),X(3,:),'--',...
      EST_EKF_SPH(1,:),EST_EKF_SPH(2,:),EST_EKF_SPH(3,:),'-',...
      S1(1),S1(2),S1(3),'x',...
      S2(1),S2(2),S2(3),'x',...
      S3(1),S3(2),S3(3),'x',...
      'LineWidth',2, 'MarkerSize',10)
legend('True trajectory','UKF estimate','Sensor 1','Sensor 2', 'Sensor 3');
title('\bf EKF SPH Solution')

xlabel('x'); ylabel('y'), zlabel('z')
axis equal
grid on
view(3)
%exportgraphics(gcf,'./figs/Real_UKF.pdf')

figure; hold on
plot(linspace(0, numSteps*dt, numSteps), P_EKF_SPH(1, :).', ...
     linspace(0, numSteps*dt, numSteps), P_EKF_SPH(2, :).', ...
     linspace(0, numSteps*dt, numSteps), P_EKF_SPH(3, :).')

plot(linspace(0, numSteps*dt, numSteps), P_EKF_SPH(4, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_EKF_SPH(5, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_EKF_SPH(6, :).', '--')
 
legend('x', 'y', 'z', 'dx', 'dy', 'dz')
xlabel('t in s')
title("P EKF SPH")


%% UKF Sph vMF
mk = x0;                % Initialize
Pk = P0;                %

EST_UKF_MF = zeros(6,numSteps);
P_UKF_MF = zeros(6,numSteps);

% Setup sigma points and weights
alpha = 0.9;
kappa = 3-6;
lambda = @(n) alpha^2 * (n+kappa) - n;
beta = 2;
W_m = @(n) [lambda(n)/(n+lambda(n)), repelem(1/(2*(n+(lambda(n)))), 2*n)];
W_c = @(n) [lambda(n)/(n+lambda(n)) + 1-alpha^2+beta, repelem(1/(2*(n+(lambda(n)))), 2*n)];

n = length(mk);
nu = sqrt(n+lambda(n));
W_m_ = W_m(n);
W_c_ = W_c(n);

MFkappa = 200;  % 200 reported in paper
% if MFkappa < 100
%     An = besseli(3/2, MFkappa) / besseli(3/2-1, MFkappa);
% else
%     An = 1 - (3-1) / (2*MFkappa);
% end
An = coth(MFkappa) - 1/MFkappa;

for k=1:numSteps
    % Prediction
    sqrtP = chol(Pk, 'lower');
    % sigma points
    sp = [mk, mk + nu*sqrtP, mk - nu*sqrtP];
    % Propagate
    Xk = A * sp;
    % Recover
    mp = sum(W_m_ .* Xk, 2);  % could write directly mp = A * mk;
    Pp = W_c_ .* (Xk-mp) * (Xk-mp).' + Q;
%     % Prediction (linear)
%     mp = A * mk;
%     Pp = A * Pk * A.' + Q;


    % Update
    % Recalculation of sigma points
    sqrtP = chol(Pp, 'lower');
    X_sp = [mp, mp + nu*sqrtP, mp - nu*sqrtP];
    % Propagate
%    z_doa = measurementFunction(sp, [S1,S2,S3]);
    Y_sp = measurementFunction(X_sp, [S1,S2,S3]);
    
    % Unit vector obeservations zi [3*j, i]
    [x_, y_, z_] = sph2cart(Y_sp(1,:), Y_sp(2,:), 1);
    zi = cat(1, x_, y_, z_);
    [x_, y_, z_] = sph2cart(Y_sp(3,:), Y_sp(4,:), 1);
    zi = cat(1, zi, x_, y_, z_);
    [x_, y_, z_] = sph2cart(Y_sp(5,:), Y_sp(6,:), 1);
    zi = cat(1, zi, x_, y_, z_);

    % TODO: PER SENSOR?
    Eh = sum(W_m(n)  .* zi, 2);
    Cxh = W_c(n) .* (X_sp-mp) * (zi - Eh).';
    Ehh = W_c(n) .* (zi) * (zi).';
    Chh = Ehh - Eh * Eh.';

    Eg = An * Eh;
    Cxg = An * Cxh;
    Cg = An^2 * Chh;
    ER = An/MFkappa * eye(9) + (1 - An^2 - 3*An/MFkappa) * Ehh;
    
    Aplus = Cxg.' / Pp;
    bplus = Eg - Aplus * mp;
    Omegaplus = Cg + ER - Aplus * Pp * Aplus.';

    % Regularize sensor data over threshold
    ovrTh = repelem(Y_diff(:, k) > diff_th, 3);  
    if any(ovrTh)
        Rp = zeros(9,9);
        R3 = R(1,1) * diag(ones(1,9)) .* (repelem(Y_diff(:, k), 3))*diff_mult;
        Rp(:, ovrTh) = R3(:, ovrTh);
        Omegaplus = Omegaplus + Rp;
    end
    % Filter
    K = Pp * Aplus.' / (Aplus*Pp*Aplus.' + Omegaplus);

    Y_k = Y(:, k);  % Current measurement
    [x_, y_, z_] = sph2cart(Y_k(1,:), Y_k(2,:), 1);
    Y_k_map = cat(1, x_, y_, z_);
    [x_, y_, z_] = sph2cart(Y_k(3,:), Y_k(4,:), 1);
    Y_k_map = cat(1, Y_k_map, x_, y_, z_);
    [x_, y_, z_] = sph2cart(Y_k(5,:), Y_k(6,:), 1);
    Y_k_map = cat(1, Y_k_map, x_, y_, z_);

%     % Why another A * mp here, but h(mp  -bplus?) 
%     mp_map = cat(1, mp(1)-S1(1), mp(2)-S1(2), mp(3)-S1(3));
%     mp_map = cat(1, mp_map, mp(1)-S2(1), mp(2)-S2(2), mp(3)-S2(3));
%     mp_map = cat(1, mp_map, mp(1)-S3(1), mp(2)-S3(2), mp(3)-S3(3));
%     v = Y_k_map - mp_map;

    v = Y_k_map - Aplus * mp - bplus;  
    mk = mp + K*v;
    Pk = Pp - K * Aplus*Pp;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    EST_UKF_MF(:,k) = mk;
    P_UKF_MF(:,k) = diag(Pk);
    
end

% Compute error
err_UKF_MF = rmse(X(1:3, :),EST_UKF_MF(1:3, :))

% Plot
figure;
plot3(X(1,:),X(2,:),X(3,:),'--',...
      EST_UKF_MF(1,:),EST_UKF_MF(2,:),EST_UKF_MF(3,:),'-',...
      S1(1),S1(2),S1(3),'x',...
      S2(1),S2(2),S2(3),'x',...
      S3(1),S3(2),S3(3),'x',...
      'LineWidth',2, 'MarkerSize',10)
legend('True trajectory','UKF estimate','Sensor 1','Sensor 2', 'Sensor 3');
title('\bf UKF SPH vMF Solution')

xlabel('x'); ylabel('y'), zlabel('z')
axis equal
grid on
view(3)
%exportgraphics(gcf,'./figs/Real_UKF.pdf')

figure; hold on
plot(linspace(0, numSteps*dt, numSteps), P_UKF_MF(1, :).', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_MF(2, :).', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_MF(3, :).')

plot(linspace(0, numSteps*dt, numSteps), P_UKF_MF(4, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_MF(5, :).', '--', ...
     linspace(0, numSteps*dt, numSteps), P_UKF_MF(6, :).', '--')
 
legend('x', 'y', 'z', 'dx', 'dy', 'dz')
xlabel('t in s')
title("P UKF vMF")


%% Evaluation
distance = @(X,EST) sqrt((sum((X-EST).^2)));

t_plot = linspace(0, numSteps*dt, numSteps);

f = figure;
f.Units = 'inches';
f.Position = [0.1 0.1 7 3];
hold on
plot(t_plot, distance(X, EST_ISC(1:3, :)), 'LineWidth',2)
plot(t_plot, distance(X, EST_UKF(1:3, :)), 'LineWidth',2)
plot(t_plot, distance(X, EST_UKF_SPH(1:3, :)), 'LineWidth',2)
%plot(t_plot, distance(X, EST_SRUKF(1:3, :)), 'LineWidth',2)
plot(t_plot, distance(X, EST_EKF_SPH(1:3, :)), 'LineWidth',2)
plot(t_plot, distance(X, EST_UKF_MF(1:3, :)), 'LineWidth',2)
ylim([0, 5])
legend("Mean Intersection", "UKF naive", "UKF SPH", "EKF SPH", "UKF vMF")
grid on
xlabel("Time in s")
ylabel("Error distance in m")

exportgraphics(gcf,'./ResultsDist.eps')

% Plot
f = figure;
f.Units = 'inches';
f.Position = [0.1 0.1 7 3];
plot3(X(1,:),X(2,:),X(3,:),'--',...
      EST_UKF(1,:),EST_UKF(2,:),EST_UKF(3,:),'-',...
      EST_UKF_SPH(1,:),EST_UKF_SPH(2,:),EST_UKF_SPH(3,:),'-',...
      EST_EKF_SPH(1,:),EST_EKF_SPH(2,:),EST_EKF_SPH(3,:),'-',...
      EST_UKF_MF(1,:),EST_UKF_MF(2,:),EST_UKF_MF(3,:),'-',...
      'LineWidth',2, 'MarkerSize',8)
hold on
plot3(S1(1),S1(2),S1(3),'ro',...
      S2(1),S2(2),S2(3),'ro',...
      S3(1),S3(2),S3(3),'ro',...
      'LineWidth',2, 'MarkerSize',8)
%text([S1(1),S2(1),S3(1)],[S1(2),S2(2),S3(2)],2+[S1(3),S2(3),S3(3)],...
%      ["1", "2", "3"], 'HorizontalAlignment','center','FontSize',10)
legend('True trajectory', ...
       'UKF naive estimate','UKF SPH estimate', 'EKF SPH estimate', 'UKF vMF estimate',...
       'Virtual microphones',...
        'Location','eastoutside');
title('\bf Filter Solution')
xlabel('x'); ylabel('y'), zlabel('z')
axis equal
grid on
xlim([0, room(1)]); ylim([0, room(2)]); zlim([0, room(3)])
view(-20, 30)
exportgraphics(gcf,'./ResultsTracking.eps','ContentType','vector')

f = figure;
f.Units = 'inches';
f.Position = [0.1 0.1 7 3];
hold on
plot(t_plot, 100*triggered_c.', 'LineWidth',2)
legend("SMA1", "SMA2", "SMA3")
grid on
xlabel("Time in s")
ylabel("\tau in %")

exportgraphics(gcf,'./tau.eps')



%%

function [intensity, energy, diff] = estimateParameters(insig)
% insig is [4 x n] B-Format.
s_pp = abs(insig(1, :)).^2;
s_vv = sum(conj(insig(2:4, :)) .* insig(2:4, :), 1);
s_pv = ones(3,1)*insig(1,:) .* conj(insig(2:4,:));
intensity = real(s_pv);
energy = 0.5 * (s_pp + s_vv);
diff = 1 - ...
    (2*sqrt(sum(real(s_pv).^2)))./(s_pp + s_vv + eps);
end


function [Ysp] = measurementFunction(Xsp, XSens)
% Xsp : sigma points [3, numSP]
% XSens : [3, numSens]
numSens = size(XSens, 2);
Ysp = zeros(2*numSens, size(Xsp, 2));  % [sens * azi/ele, numSigmapoints]
for idxSens = 1:numSens
    [azi_, ele_, ~] = cart2sph(Xsp(1,:) - XSens(1,idxSens),...
                              Xsp(2,:) - XSens(2,idxSens),...
                              Xsp(3,:) - XSens(3,idxSens));
    Ysp(idxSens*2-1, :) = azi_;
    Ysp(idxSens*2,:) = ele_;
end
end


function [p_isc] = doaIntersect(pS1, doa1, pS2, doa2)
% cartesian position sensor, spherical doa
% intersection position cartesian
    [dirx, diry, dirz] = sph2cart(doa1(1), doa1(2), 1);
    dir1 = [dirx, diry, dirz].';
    [dirx, diry, dirz] = sph2cart(doa2(1), doa2(2), 1);
    dir2 = [dirx, diry, dirz].';    

    tau1 = ((pS2-pS1).'* dir1 + (pS1-pS2).' * dir2*(dir1.'*dir2)) / ...
            (1- (dir1.'*dir2)^2);
    tau2 = ((pS1-pS2).'* dir2 + (pS2-pS1).' * dir1*(dir1.'*dir2)) / ...
            (1- (dir1.'*dir2)^2);
    
    % Filter out negative
    if (sign(tau1) + sign(tau2)) < 2
        tau1 = NaN;
        tau2 = NaN;
    end
        
    p_isc = (pS1 + tau1*dir1 + pS2 + tau2*dir2)./ 2;

end


function [XYZ] = sph2cartVec(azi, ele, r)
[xi, yi, zi] = sph2cart(azi, ele, r);
XYZ = cat(1, xi, yi, zi);
end



